"""Survival-first mechfight bot.

Priorities: stay inside the zone (nav field around buildings), fight back with leading
gun/lightning/laser while strafing, grab cheap loot (drones, crates, cubes, weapons, supply drops)
only when nothing hostile is in sight, face whoever hurts us (rear armor is x1.5).
All tunables live in CFG.
"""
import array
import heapq
import json
import math
import random
import sys
import threading
import time

import mechfight as mf

CFG = dict(
    cell=4.0,
    zone_margin=30.0,       # stay this far inside the circle we are heading for
    loot_radius=160.0,      # max detour for loot when nothing hostile is visible
    drop_radius=220.0,      # go for a supply drop if this close
    pref_gun=70.0, pref_laser=90.0, pref_lightning=28.0,
    strafe_min=0.35, strafe_max=1.2,
    retreat_enter=0.0,      # start retreating below this HP fraction
    retreat_exit=0.80,      # resume normal play above this
    retreat_max=25.0,       # give up retreating after this many seconds
    cornered_dist=28.0,     # an enemy this close: fight instead of running
    hold_radius=18.0,       # stand still (silent) this close to the goal when idle
)
PICK_ORDER = ["armor", "plating", "nanites", "gun_damage", "gun_rate", "servos", "gyros", "reactor",
              "capacitor", "gun_mag", "jets", "sensors", "gun_velocity"]
WEAPON_RANK = {"lightning": 3, "laser": 2, "mortar": 0, "gun": 1}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def norm(vx, vy, cap=1.0):
    n = math.hypot(vx, vy)
    if n > cap:
        return vx / n * cap, vy / n * cap
    return vx, vy


class Nav:
    """Coarse grid of the map; a distance-to-goal field is computed in a worker thread."""

    def __init__(self, mapdir):
        c = CFG["cell"]
        self.c = c
        t = None
        w = json.load(open(f"{mapdir}/v2/world.json"))
        t = w["terrain"]
        self.tx, self.ty, self.tn, self.tm = t["minX"], t["minY"], t["nx"], t["ny"]
        raw = array.array("h")
        with open(f"{mapdir}/v2/{t['file']}", "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder != "little":
            raw.byteswap()
        self.h = raw
        self.x0, self.y0 = self.tx, self.ty
        self.gw = int((self.tn - 1) / c)
        self.gh = int((self.tm - 1) / c)
        self.blocked = bytearray(self.gw * self.gh)
        self.bld = bytearray(self.gw * self.gh)
        self._raster(w["prisms"])
        self._slopes()
        self.dist = None
        self.goal_key = None
        self.want = None
        self.ev = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def ground(self, x, y):
        i = min(max(int(round(x - self.tx)), 0), self.tn - 1)
        j = min(max(int(round(y - self.ty)), 0), self.tm - 1)
        return self.h[j * self.tn + i] * 0.01

    def _raster(self, prisms):
        c, gw, gh = self.c, self.gw, self.gh
        offs = [(0, 0), (2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5)]
        for p in prisms:
            if p["kind"] == "bridge":
                continue
            ring = p["ring"]
            xs, ys = ring[0::2], ring[1::2]
            rings = [(xs, ys)] + [(h[0::2], h[1::2]) for h in p.get("holes") or []]
            gx0 = max(int((min(xs) - 3 - self.x0) / c), 0)
            gx1 = min(int((max(xs) + 3 - self.x0) / c) + 1, gw - 1)
            gy0 = max(int((min(ys) - 3 - self.y0) / c), 0)
            gy1 = min(int((max(ys) + 3 - self.y0) / c) + 1, gh - 1)
            base = p["base"]
            for gy in range(gy0, gy1 + 1):
                for gx in range(gx0, gx1 + 1):
                    if self.blocked[gy * gw + gx]:
                        continue
                    cx, cy = self.x0 + (gx + .5) * c, self.y0 + (gy + .5) * c
                    if base - self.ground(cx, cy) > 9.5:
                        continue
                    for dx, dy in offs:
                        px, py = cx + dx, cy + dy
                        if self._inside(px, py, rings[0]) and not any(self._inside(px, py, r) for r in rings[1:]):
                            self.blocked[gy * gw + gx] = 1
                            self.bld[gy * gw + gx] = 1
                            break

    @staticmethod
    def _inside(px, py, ring):
        xs, ys = ring
        n, ins, j = len(xs), False, len(xs) - 1
        for i in range(n):
            if (ys[i] > py) != (ys[j] > py) and px < (xs[j] - xs[i]) * (py - ys[i]) / (ys[j] - ys[i]) + xs[i]:
                ins = not ins
            j = i
        return ins

    def _slopes(self):
        c, gw, gh = self.c, self.gw, self.gh
        for gy in range(gh):
            for gx in range(gw):
                x, y = self.x0 + (gx + .5) * c, self.y0 + (gy + .5) * c
                h0 = self.ground(x, y)
                for dx, dy in ((c, 0), (-c, 0), (0, c), (0, -c)):
                    if abs(self.ground(x + dx, y + dy) - h0) / c > 1.1:
                        self.blocked[gy * gw + gx] = 1
                        break

    def clear(self, x0, y0, x1, y1):
        """True if no building cell lies on the straight line (2 m steps)."""
        d = math.hypot(x1 - x0, y1 - y0)
        n = max(int(d / 2.0), 1)
        for k in range(1, n):
            f = k / n
            if self.bld[self.cell(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)]:
                return False
        return True

    def cell(self, x, y):
        return (min(max(int((y - self.y0) / self.c), 0), self.gh - 1) * self.gw
                + min(max(int((x - self.x0) / self.c), 0), self.gw - 1))

    def center(self, idx):
        gy, gx = divmod(idx, self.gw)
        return self.x0 + (gx + .5) * self.c, self.y0 + (gy + .5) * self.c

    def request(self, x, y):
        key = self.cell(x, y)
        if key != self.goal_key and key != self.want:
            self.want = key
            self.ev.set()

    def _worker(self):
        while True:
            self.ev.wait()
            self.ev.clear()
            goal = self.want
            try:
                self.dist = self._dijkstra(goal)
                self.goal_key = goal
            except Exception as e:  # never kill the worker
                print("nav error", e, file=sys.stderr)

    def _dijkstra(self, goal):
        gw, gh, blk = self.gw, self.gh, self.blocked
        INF = 1e18
        d = [INF] * (gw * gh)
        if blk[goal]:  # snap to nearest free cell
            seen, fr = {goal}, [goal]
            while fr and blk[goal]:
                nxt = []
                for i in fr:
                    gy, gx = divmod(i, gw)
                    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = gx + ox, gy + oy
                        if 0 <= nx < gw and 0 <= ny < gh and (ny * gw + nx) not in seen:
                            seen.add(ny * gw + nx)
                            nxt.append(ny * gw + nx)
                            if not blk[ny * gw + nx]:
                                goal = ny * gw + nx
                                break
                    if not blk[goal]:
                        break
                fr = nxt
        d[goal] = 0.0
        pq = [(0.0, goal)]
        nb = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
        while pq:
            dd, i = heapq.heappop(pq)
            if dd > d[i]:
                continue
            gy, gx = divmod(i, gw)
            for ox, oy, cost in nb:
                nx, ny = gx + ox, gy + oy
                if not (0 <= nx < gw and 0 <= ny < gh):
                    continue
                j = ny * gw + nx
                if blk[j]:
                    continue
                if ox and oy and (blk[gy * gw + nx] or blk[ny * gw + gx]):
                    continue
                nd = dd + cost
                if nd < d[j]:
                    d[j] = nd
                    heapq.heappush(pq, (nd, j))
        return d

    def step(self, x, y, lookahead=3):
        """Waypoint (x, y) `lookahead` cells downhill of the field, or None if no path."""
        d = self.dist
        if d is None:
            return None
        gw, gh = self.gw, self.gh
        i = self.cell(x, y)
        INF = 1e17
        for _ in range(lookahead):
            gy, gx = divmod(i, gw)
            best, bi = d[i], i
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nx, ny = gx + ox, gy + oy
                    if (ox or oy) and 0 <= nx < gw and 0 <= ny < gh:
                        j = ny * gw + nx
                        if d[j] < best:
                            best, bi = d[j], j
            if bi == i:
                break
            i = bi
        if d[i] > INF:
            return None
        return self.center(i)


def lead(origin, tpos, tvel, speed, iters=3):
    px, py, pz = tpos
    t = 0.0
    for _ in range(iters):
        dist = math.dist((px, py, pz), origin)
        t = dist / speed
        px, py, pz = tpos[0] + tvel[0] * t, tpos[1] + tvel[1] * t, tpos[2] + tvel[2] * t
    return px, py, pz


class Bot:
    name = "survivor-v1"

    def __init__(self, hello):
        cfg = hello["config"]
        self.cfg = cfg
        self.rng = random.Random(hello.get("seed", 0))
        self.nav = Nav(hello["map"]["dir"])
        self.sdir = 1
        self.s_until = 0.0
        self.hurt_bearing = None
        self.hurt_t = -99.0
        self.blacklist = {}
        self.stuck_ref = None
        self.last_target_kind = None
        self.retreat = False
        self.retreat_start = 0.0
        self.retreat_cd = 0.0
        self.cover = None
        self.cover_t = -99.0
        sys.setswitchinterval(0.001)

    # ---- helpers -------------------------------------------------
    def zone_info(self, obs, me_xy):
        z = obs["zone"]
        use_next = obs["t"] >= z["shrink_start"] - 12
        cx, cy = z["next_center"] if use_next else z["center"]
        r = z["next_radius"] if use_next else z["radius"]
        d = math.dist(me_xy, (cx, cy))
        return (cx, cy), r, d

    def pick_offer(self, offer):
        best, bi = 99, 0
        for i, ch in enumerate(offer["choices"]):
            r = PICK_ORDER.index(ch["id"]) if ch["id"] in PICK_ORDER else 50
            if r < best:
                best, bi = r, i
        return bi

    def act(self, obs):
        t = obs["t"]
        me = obs["self"]
        x, y, z = me["pos"]
        stats = me["stats"]
        act = {"tick": obs["tick"]}
        if obs.get("offer"):
            act["pick"] = self.pick_offer(obs["offer"])

        weapons = me["weapons"]
        kinds = [w["kind"] if w else None for w in weapons]
        yaw_now, pit_now = me["aim"]
        muzzle = (x, y, z + stats.get("muzzle_height", 7))

        for ev in obs.get("events", []):
            if ev["type"] == "damaged" and ev.get("bearing") is not None and ev.get("weapon") != "zone":
                self.hurt_bearing, self.hurt_t = ev["bearing"], t

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        mechs.sort(key=lambda m: math.dist(m["pos"], me["pos"]) + m.get("hp", 1000) / 25.0)
        # ---- zone / goal ----
        (zcx, zcy), zr, zd = self.zone_info(obs, (x, y))
        in_target = zd < zr - CFG["zone_margin"]
        goal = list(obs["zone"]["next_center"])
        for dr in obs.get("drops", []):
            dd = math.dist((x, y), dr["pos"][:2])
            if dr["holder"] in (None, me["id"]) and dd < CFG["drop_radius"] and dr["lands_in"] < dd / 12 + 8 \
                    and not mechs:
                goal = dr["pos"][:2]
        self.nav.request(goal[0], goal[1])
        wp = self.nav.step(x, y)
        gdist = math.dist((x, y), goal)
        if wp is not None and gdist > 12:
            nvx, nvy = wp[0] - x, wp[1] - y
        else:
            nvx, nvy = goal[0] - x, goal[1] - y
        nvx, nvy = norm(nvx, nvy, 1.0)
        n = math.hypot(nvx, nvy)
        if n > 1e-6:
            nvx, nvy = nvx / n, nvy / n

        # ---- retreat to heal (regen needs 5 s without dealing or taking damage) ----
        hpf = me["hp"] / me["max_hp"]
        threats = [m["pos"][:2] for m in mechs if math.dist(m["pos"], me["pos"]) < 220]
        hurt_now = any(e["type"] == "damaged" and e.get("weapon") != "zone" for e in obs.get("events", []))
        if not threats and t - self.hurt_t < 3.0 and self.hurt_bearing is not None:
            threats = [(x + 60 * math.cos(self.hurt_bearing), y + 60 * math.sin(self.hurt_bearing))]
        close = bool(mechs) and math.dist(mechs[0]["pos"], me["pos"]) < CFG["cornered_dist"]
        if self.retreat:
            if hpf > CFG["retreat_exit"] or t - self.retreat_start > CFG["retreat_max"] or not me["in_zone"]:
                self.retreat, self.retreat_cd = False, t + 8.0
        elif hpf < CFG["retreat_enter"] and threats and me["in_zone"] and t >= self.retreat_cd:
            self.retreat, self.retreat_start, self.cover = True, t, None
        if self.retreat and threats and not close:
            r = self.do_retreat(obs, me, act, threats, hurt_now, t, (zcx, zcy), zr, (nvx, nvy), weapons)
            if r is not None:
                return r

        # ---- pick a target ----
        target, tkind = None, None
        if mechs:
            target, tkind = mechs[0], "mech"
        else:
            cands = []
            for m in npcs:
                cands.append((math.dist(m["pos"], me["pos"]), m, "npc"))
            for c in obs["crates"]:
                cands.append((math.dist(c["pos"], me["pos"]) + 10, dict(c, vel=[0, 0, 0]), "crate"))
            cands = [c for c in cands if c[0] < CFG["loot_radius"]
                     and self.blacklist.get(c[1]["id"], -1) < t]
            if cands and in_target and me["hp"] > 0.5 * me["max_hp"]:
                cands.sort(key=lambda c: c[0])
                _, target, tkind = cands[0]

        mvx, mvy = nvx, nvy
        aim = None
        fire = [False, False]
        boost = False
        speed = stats.get("walk_speed", 11.7)

        if target is not None:
            tp = target["pos"]
            tc = (tp[0], tp[1], tp[2] + (4.5 if tkind != "npc" else 1.7))
            d = math.dist(tc, muzzle)
            gun_i = kinds.index("gun") if "gun" in kinds else None
            gstat = weapons[gun_i]["stats"] if gun_i is not None else {"speed": 252, "range": 230}
            lp = lead(muzzle, tc, target.get("vel", [0, 0, 0]), gstat["speed"])
            ayaw = math.atan2(lp[1] - muzzle[1], lp[0] - muzzle[0])
            hd = math.hypot(lp[0] - muzzle[0], lp[1] - muzzle[1])
            apit = math.atan2(lp[2] - muzzle[2], hd)
            apit = min(max(apit, stats.get("min_pitch", -0.7)), stats.get("max_pitch", 1.3))
            aim = [ayaw, apit]
            yerr = abs(wrap(ayaw - yaw_now))
            perr = abs(apit - pit_now)
            thr = max(0.02, math.atan(1.8 / max(d, 1)))
            aligned = yerr < thr and perr < thr * 1.5

            if gun_i is not None and d < gstat["range"] * 0.92 and aligned:
                fire[gun_i] = True
            if "gun" in kinds and weapons[kinds.index("gun")]["ammo"] == 0:
                act["reload"] = True
            for i, k in enumerate(kinds):
                w = weapons[i]
                if k == "lightning" and d < 42 and yerr < 0.35 and me["energy"] > 6:
                    fire[i] = True
                elif k == "laser" and d < w["stats"].get("range", 280) * 0.9:
                    full = w["stats"].get("charge", 0.75)
                    if w["cooldown"] > 0.05:
                        fire[i] = False
                    elif w.get("charge", 0) >= full - 0.06:
                        fire[i] = not aligned  # release when lined up
                    else:
                        fire[i] = yerr < 0.25 or w.get("charge", 0) > 0 and yerr < 0.5

            # ---- movement while fighting ----
            if tkind == "mech":
                pref = CFG["pref_gun"]
                if "lightning" in kinds:
                    pref = CFG["pref_lightning"]
                elif "laser" in kinds:
                    pref = CFG["pref_laser"]
                if t >= self.s_until:
                    self.sdir = -self.sdir if self.rng.random() < 0.7 else self.sdir
                    self.s_until = t + self.rng.uniform(CFG["strafe_min"], CFG["strafe_max"])
                b = math.atan2(tc[1] - y, tc[0] - x)
                rad = max(-1.0, min(1.0, (d - pref) / 25.0))
                sx, sy = -math.sin(b) * self.sdir, math.cos(b) * self.sdir
                navw = 0.15 if in_target else 1.0
                mvx = sx * 0.9 + math.cos(b) * rad * 0.7 + nvx * navw
                mvy = sy * 0.9 + math.sin(b) * rad * 0.7 + nvy * navw
            else:
                # drone / crate: close to a comfortable range then hold
                b = math.atan2(tc[1] - y, tc[0] - x)
                want = 30.0 if tkind == "crate" else 38.0
                if d > want + 5:
                    mvx, mvy = math.cos(b), math.sin(b)
                elif d < want - 8 and tkind == "npc":
                    mvx, mvy = -math.cos(b) * 0.5, -math.sin(b) * 0.5
                else:
                    mvx = mvy = 0.0
                if self.stuck(t, x, y):
                    self.blacklist[target["id"]] = t + 20
        else:
            # ---- idle: loot or head for goal ----
            loot = self.pick_loot(obs, me, kinds, in_target, t)
            if loot is not None:
                lx, ly = loot["pos"][:2]
                dd = math.dist((x, y), (lx, ly))
                mvx, mvy = norm(lx - x, ly - y, 1.0)
                if dd > 1e-6:
                    mvx, mvy = (lx - x) / dd, (ly - y) / dd
                if self.stuck(t, x, y):
                    self.blacklist[loot["id"]] = t + 20
                if loot.get("kind") in ("mortar", "laser", "lightning") and dd < 3.5 and abs(loot["pos"][2] - z) < 4:
                    act["take"] = loot["id"]
                    if weapons[1] is not None and kinds[1] != loot["kind"]:
                        act["drop"] = 1
            elif gdist < CFG["hold_radius"] and in_target:
                mvx = mvy = 0.0
            if self.hurt_bearing is not None and t - self.hurt_t < 2.0:
                aim = [self.hurt_bearing, 0.0]
                fire[0] = kinds[0] == "gun"  # blind fire at the attacker
            else:
                aim = self.sound_aim(obs, x, y) or ([math.atan2(mvy, mvx), 0.0] if abs(mvx) + abs(mvy) > .1 else None)

        # ---- projectile dodge ----
        dx, dy = self.dodge(obs, x, y, z)
        mvx, mvy = mvx + dx, mvy + dy

        # ---- zone safety overrides ----
        if not me["in_zone"]:
            mvx, mvy = nvx, nvy
            boost = me["energy"] > 15
        elif zd > zr - 8:
            mvx, mvy = mvx * 0.5 + nvx, mvy * 0.5 + nvy
        elif gdist > 220 and not mechs and me["energy"] > 55:
            boost = True
        mvx, mvy = norm(mvx, mvy, 1.0)

        act.update(move=[mvx, mvy], boost=boost, fire=fire)
        if aim is not None:
            act["aim"] = aim
        # a held weapon whose slot isn't "fire"-eligible this tick still needs an explicit False for lasers
        return act

    def do_retreat(self, obs, me, act, threats, hurt_now, t, zc, zr, nav_dir, weapons):
        x, y, z = me["pos"]
        nav = self.nav
        tx = sum(p[0] for p in threats) / len(threats)
        ty = sum(p[1] for p in threats) / len(threats)
        hidden_here = all(not nav.clear(p[0], p[1], x, y) for p in threats)
        if hidden_here:
            self.cover, self.cover_t = (x, y), t
        elif self.cover is None or hurt_now or t - self.cover_t > 1.5:
            best, bc = None, 1e18
            for r in (14, 24, 36, 48):
                for k in range(16):
                    a = k * math.pi / 8
                    cx, cy = x + r * math.cos(a), y + r * math.sin(a)
                    if nav.blocked[nav.cell(cx, cy)] or math.dist((cx, cy), zc) > zr - 25:
                        continue
                    if not nav.clear(x, y, cx, cy):
                        continue
                    if any(nav.clear(p[0], p[1], cx, cy) for p in threats):
                        continue
                    cost = r - 0.3 * math.dist((cx, cy), (tx, ty))
                    if cost < bc:
                        best, bc = (cx, cy), cost
            self.cover, self.cover_t = best, t
        if self.cover is not None:
            dx, dy = self.cover[0] - x, self.cover[1] - y
            dd = math.hypot(dx, dy)
            mv = (0.0, 0.0) if dd < 2.0 else (dx / dd, dy / dd)
        else:  # nowhere to hide: run away from the threats, drifting toward the safe zone
            ax, ay = x - tx, y - ty
            an = math.hypot(ax, ay) or 1.0
            mv = norm(ax / an + nav_dir[0] * 0.6, ay / an + nav_dir[1] * 0.6, 1.0)
        ox, oy = self.dodge(obs, x, y, z)
        mvx, mvy = norm(mv[0] + ox, mv[1] + oy, 1.0)
        act.update(move=[mvx, mvy], fire=[False, False],
                   boost=bool((mvx or mvy) and me["energy"] > 40 and not hidden_here),
                   aim=[math.atan2(ty - y, tx - x), 0.0])
        for i, w in enumerate(weapons):
            if w and w["kind"] == "gun" and w["ammo"] < w["stats"].get("mag", 12) / 2 and w["reload"] == 0:
                act["reload"] = True
        return act

    # ---- sub-behaviours --------------------------------------------
    def stuck(self, t, x, y):
        if self.stuck_ref is None or math.dist(self.stuck_ref[1:], (x, y)) > 3.0:
            self.stuck_ref = (t, x, y)
            return False
        return t - self.stuck_ref[0] > 3.0

    def pick_loot(self, obs, me, kinds, in_target, t):
        if not in_target:
            return None
        x, y, _ = me["pos"]
        best, bd = None, CFG["loot_radius"]
        for p in obs["pickups"]:
            k = p["kind"]
            held = [WEAPON_RANK.get(q, 0) for q in kinds[1:] if q]
            if k in kinds or WEAPON_RANK.get(k, 0) <= 1:
                continue
            if kinds[1] is not None and WEAPON_RANK.get(k, 0) <= WEAPON_RANK.get(kinds[1], 0):
                continue
            if self.blacklist.get(p["id"], -1) > t:
                continue
            d = math.dist((x, y), p["pos"][:2]) - 15
            if d < bd:
                best, bd = p, d
        for c in obs["cubes"]:
            if self.blacklist.get(c["id"], -1) > t:
                continue
            d = math.dist((x, y), c["pos"][:2])
            if d < bd:
                best, bd = dict(c, kind="cube"), d
        return best

    def sound_aim(self, obs, x, y):
        best = None
        for s in obs.get("sounds", []):
            if s["kind"] in ("gun", "laser", "lightning", "mortar", "explosion") and s["dist"] in ("near", "mid"):
                if s["age"] < 0.6 and (best is None or s["dist"] == "near"):
                    best = s
        if best is not None:
            return [best["bearing"], 0.0]
        return None

    def dodge(self, obs, x, y, z):
        ox = oy = 0.0
        for p in obs.get("projectiles", []):
            px, py, pz = p["pos"][0] - x, p["pos"][1] - y, p["pos"][2] - z - 4.5
            vx, vy, vz = p["vel"]
            v2 = vx * vx + vy * vy + vz * vz
            if v2 < 1:
                continue
            tt = -(px * vx + py * vy + pz * vz) / v2
            if 0 < tt < 0.7:
                cx, cy = px + vx * tt, py + vy * tt
                if math.hypot(cx, cy) < 4.5:
                    vn = math.hypot(vx, vy) or 1.0
                    perp = (-vy / vn, vx / vn)
                    side = 1.0 if (cx * perp[0] + cy * perp[1]) <= 0 else -1.0
                    ox += perp[0] * side * 1.2
                    oy += perp[1] * side * 1.2
        return ox, oy


mf.run(Bot)
