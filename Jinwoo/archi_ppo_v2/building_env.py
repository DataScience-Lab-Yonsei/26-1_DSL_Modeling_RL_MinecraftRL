"""
House Building Env for CraftGround v2.6.15 — v6

Stage 0 "Place Any Block":
  - Task: place ANY block at the correct blueprint position (block type ignored)
  - Inventory: only minecraft:dirt (no hotbar management needed)
  - Completion = all blueprint positions filled with any block
  - Simpler stepping stone before learning correct block types

Stage 1+: full building (block type still position-only — extend for type checking later).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum

from nbt_parser import Blueprint, BlueprintBlock

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Actions ───────────────────────────────────────────────────────
class BuildAction(IntEnum):
    NO_OP        = 0
    FORWARD      = 1
    BACKWARD     = 2
    STRAFE_LEFT  = 3
    STRAFE_RIGHT = 4
    JUMP         = 5
    LOOK_UP      = 6
    LOOK_DOWN    = 7
    LOOK_LEFT    = 8
    LOOK_RIGHT   = 9
    PLACE_BLOCK  = 10
    BREAK_BLOCK  = 11
    JUMP_AND_PLACE = 12
    HOTBAR_1     = 13
    HOTBAR_2     = 14
    HOTBAR_3     = 15

NUM_BUILD_ACTIONS = len(BuildAction)


CAMERA_PITCH_STEP = 10.0
CAMERA_YAW_STEP   = 15.0


def _noop_action():
    return {
        'forward':0,'back':0,'left':0,'right':0,
        'jump':0,'sneak':0,'sprint':0,
        'attack':0,'use':0,'drop':0,'inventory':0,
        'camera': np.array([0.0,0.0], dtype=np.float32),
        'hotbar.1':0,'hotbar.2':0,'hotbar.3':0,
        'hotbar.4':0,'hotbar.5':0,'hotbar.6':0,
        'hotbar.7':0,'hotbar.8':0,'hotbar.9':0,
    }

def build_action_to_v2(action):
    a = _noop_action()
    if   action == BuildAction.FORWARD:      a['forward']=1
    elif action == BuildAction.BACKWARD:     a['back']=1
    elif action == BuildAction.STRAFE_LEFT:  a['left']=1
    elif action == BuildAction.STRAFE_RIGHT: a['right']=1
    elif action == BuildAction.JUMP:         a['jump']=1
    elif action == BuildAction.LOOK_UP:
        a['camera']=np.array([-CAMERA_PITCH_STEP,0.0],dtype=np.float32)
    elif action == BuildAction.LOOK_DOWN:
        a['camera']=np.array([CAMERA_PITCH_STEP,0.0],dtype=np.float32)
    elif action == BuildAction.LOOK_LEFT:
        a['camera']=np.array([0.0,-CAMERA_YAW_STEP],dtype=np.float32)
    elif action == BuildAction.LOOK_RIGHT:
        a['camera']=np.array([0.0,CAMERA_YAW_STEP],dtype=np.float32)
    elif action == BuildAction.PLACE_BLOCK:  a['use']=1
    elif action == BuildAction.BREAK_BLOCK:  a['attack']=1
    elif action == BuildAction.JUMP_AND_PLACE:
        a['jump']=1; a['use']=1
        a['camera']=np.array([CAMERA_PITCH_STEP*3,0.0],dtype=np.float32)
    elif action == BuildAction.HOTBAR_1: a['hotbar.1']=1
    elif action == BuildAction.HOTBAR_2: a['hotbar.2']=1
    elif action == BuildAction.HOTBAR_3: a['hotbar.3']=1
    return a


@dataclass
class BuildingConfig:
    blueprint: Optional[Blueprint] = None
    build_origin: Tuple[int,int,int] = (0,-60,0)
    local_obs_radius: int = 5
    image_width: int = 64
    image_height: int = 64
    use_visual_obs: bool = False
    # Building rewards
    reward_correct_place: float = 1.0
    reward_partial_place: float = 0.3
    reward_wrong_place: float = -0.5
    reward_scaffold_cleanup: float = 0.5
    reward_layer_complete: float = 10.0
    reward_structure_complete: float = 100.0
    reward_approach_target: float = 0.02   # delta-based: kept but reduced
    penalty_time_step: float = -0.002
    penalty_fall: float = -0.1
    penalty_death: float = -5.0
    penalty_remaining_scaffold: float = -0.2
    height_reward_scale: float = 1.0
    reward_in_order: float = 0.2
    penalty_out_of_order: float = -0.1
    # Dense guidance rewards
    reward_proximity_max: float = 0.008   # per-step within 5 blocks, scaled by closeness
    reward_proximity_range: float = 5.0
    reward_aim_correct: float = 0.5       # raycast would place exactly at target
    reward_near_miss: float = 0.1         # placement within 1 block (Manhattan) of target
    reward_aim_angle: float = 0.015       # per-step, scaled linearly from 0 at threshold to max at 0°
    reward_aim_angle_threshold: float = 20.0  # degrees — within this: reward; beyond: penalty
    penalty_aim_angle: float = -0.005     # per-step penalty when angle > threshold, scales with overshoot
    reward_aim_on_target: float = 0.02    # per-step bonus when raycast confirms hit on block A top face
    aim_streak_cap: int = 5              # taper aim_on_target to 0 after this many consecutive steps
    # aim exit penalty = clawback (exactly cancels accumulated aim streak reward)
    # Place gate: suppress PLACE when aim error exceeds this threshold (0 = disabled)
    place_aim_gate_threshold: float = 25.0
    reward_place_bad_aim: float = 0.0     # no penalty for gated PLACE — suppression alone is enough
    # General
    max_timesteps: int = 2000
    allow_scaffold: bool = True
    curriculum_stage: int = 0
    structure_name: str = ""   # e.g. "row", "row_2", "pillar_2", "wall_3high", …
    # Blueprint generator — called on every reset() to get a new blueprint
    # (used for random blueprint generation each episode)
    blueprint_generator: Optional[object] = None  # callable() -> Blueprint


# ── parse ─────────────────────────────────────────────────────────
def _parse_full_obs(obs_dict):
    full = obs_dict.get('full')
    r = {'x':0.0,'y':-60.0,'z':0.0,'yaw':0.0,'pitch':0.0,
         'health':20.0,'is_dead':False,'image':None,
         'raycast_type':'MISS','raycast_block_x':0,'raycast_block_y':0,
         'raycast_block_z':0,'raycast_block_name':'','raycast_face':''}
    if full is not None:
        for k in ('x','y','z','yaw','pitch','health'):
            r[k] = float(getattr(full,k,r[k]))
        r['is_dead'] = bool(getattr(full,'is_dead',False))
        rc = getattr(full,'raycast_result',None)
        if rc:
            t = getattr(rc,'type',None)
            if t is not None and ('BLOCK' in str(t).upper() or str(t)=='1'):
                r['raycast_type']='BLOCK'
                tb = getattr(rc,'target_block',None)
                if tb:
                    r['raycast_block_x']=int(getattr(tb,'x',0))
                    r['raycast_block_y']=int(getattr(tb,'y',0))
                    r['raycast_block_z']=int(getattr(tb,'z',0))
                    r['raycast_block_name']=str(getattr(tb,'translation_key',''))
                face = getattr(rc,'face',getattr(rc,'target_side',None))
                if face is not None: r['raycast_face']=str(face)
    for key in ('pov','rgb'):
        v = obs_dict.get(key)
        if v is not None and isinstance(v, np.ndarray):
            r['image']=v; break
    return r

def _compute_place_position(parsed, ax, az):
    if parsed['raycast_type']!='BLOCK': return None
    hx,hy,hz = parsed['raycast_block_x'],parsed['raycast_block_y'],parsed['raycast_block_z']
    face = str(parsed.get('raycast_face','')).upper()
    if 'UP' in face or 'TOP' in face:      return (hx,hy+1,hz)
    if 'DOWN' in face or 'BOTTOM' in face:  return (hx,hy-1,hz)
    if 'NORTH' in face: return (hx,hy,hz-1)
    if 'SOUTH' in face: return (hx,hy,hz+1)
    if 'EAST' in face:  return (hx+1,hy,hz)
    if 'WEST' in face:  return (hx-1,hy,hz)
    pitch = parsed['pitch']
    if pitch>45:  return (hx,hy+1,hz)
    if pitch<-45: return (hx,hy-1,hz)
    dx=ax-(hx+0.5); dz=az-(hz+0.5)
    if abs(dx)>abs(dz): return (hx+(1 if dx>0 else -1),hy,hz)
    return (hx,hy,hz+(1 if dz>0 else -1))

def _angle_to_target(ax,ay,az,yaw,pitch,tx,ty,tz):
    dx,dz=tx-ax,tz-az; dy=ty-(ay+1.62)
    dh=math.sqrt(dx**2+dz**2)
    t_yaw=math.degrees(math.atan2(-dx,dz))
    t_pitch=math.degrees(math.atan2(-dy,max(dh,0.01)))
    yd=abs(((yaw-t_yaw+180)%360)-180); pd=abs(pitch-t_pitch)
    return yd,pd,math.sqrt(yd**2+pd**2)


# ══════════════════════════════════════════════════════════════════
class HouseBuildingWrapper(gym.Wrapper):

    def __init__(self, env, config, debug_visual=False):
        super().__init__(env)
        self.config = config
        self.blueprint = config.blueprint
        self.debug_visual = debug_visual
        obs_size = 2*config.local_obs_radius+1
        self.obs_size = obs_size

        obs_dict = {
            # Agent knows: position, yaw, pitch, health
            "agent_pos": spaces.Box(
                np.array([-1000,-64,-1000,-180,-90,0],dtype=np.float32),
                np.array([1000,320,1000,180,90,20],dtype=np.float32)),
            "local_grid":  spaces.Box(0,255,(obs_size,)*3,np.uint8),
            "target_grid": spaces.Box(0,255,(obs_size,)*3,np.uint8),
            "diff_grid":   spaces.Box(-1,1,(obs_size,)*3,np.int8),
            "raycast_grid": spaces.Box(0,1,(obs_size,)*3,np.float32),
            "progress":    spaces.Box(0,10000,(4,),np.float32),
            "next_block_id": spaces.Box(0.0, 1.0, (1,), np.float32),
            # Structure info: [type_id, n_blocks, size_x, size_y, size_z] normalised.
            # type_id: 0=single, 0.25=row, 0.5=wall, 0.75=room, 1.0=full
            "structure_info": spaces.Box(0.0, 1.0, (5,), np.float32),
        }
        if config.use_visual_obs:
            obs_dict["image"]=spaces.Box(0,255,(config.image_height,config.image_width,3),np.uint8)
        self.observation_space = spaces.Dict(obs_dict)

        # Stage 0 removed — full action space from the start
        self.action_space = spaces.Discrete(NUM_BUILD_ACTIONS)

        self._world_blocks = {}
        self._scaffold_blocks = set()
        self._correctly_placed = set()
        self._step_count = 0
        self._prev_agent_pos = None
        self._total_reward = 0.0
        self._last_info = ""
        self._pre_step_parsed = None
        self._current_hotbar = 1  # tracks active hotbar slot (1-indexed)
        self._aim_streak = 0           # consecutive steps raycast is on block A
        self._aim_streak_reward = 0.0  # cumulative aim_on_target reward in streak
        self._was_on_target = False    # whether previous step was on block A
        self._rc: dict = {}  # per-episode reward component accumulators
        self._reset_rc()
        self._structure_info = self._compute_structure_info()

    # ── reset ─────────────────────────────────────────────────────
    def reset(self, **kwargs):
        # Regenerate blueprint on each reset if generator is set
        if self.config.blueprint_generator is not None:
            self.blueprint = self.config.blueprint_generator()
            self.config.blueprint = self.blueprint
            self._structure_info = self._compute_structure_info()

        obs_raw, info = self.env.reset(**kwargs)
        self._world_blocks.clear(); self._scaffold_blocks.clear()
        self._correctly_placed.clear()
        self._step_count=0; self._prev_agent_pos=None; self._total_reward=0.0
        self._last_info=""
        self._pre_step_parsed=None
        self._current_hotbar = 1
        self._aim_streak = 0
        self._aim_streak_reward = 0.0
        self._was_on_target = False
        self._reset_rc()

        ox,oy,oz = self.config.build_origin
        sx=max(self.blueprint.size_x,1)
        sy=max(self.blueprint.size_y,1)
        sz=max(self.blueprint.size_z,1)

        # ── Build a stone platform under the build area + spawn zone ──
        # Margin around the blueprint so agent can walk around it
        margin = 5
        # Agent spawns at tz+3 from first target, so extend +Z by extra
        spawn_margin_z = 6
        px1, px2 = ox - margin, ox + sx + margin
        pz1, pz2 = oz - margin, oz + sz + spawn_margin_z
        # Platform: 3-block-thick stone slab just below build origin
        platform_top = oy - 1
        platform_bot = oy - 3
        # Clear air above the platform for headroom (build height + walking space)
        clear_top = oy + sy + 10

        platform_cmds = [
            # Stone platform
            f"fill {px1} {platform_bot} {pz1} {px2} {platform_top} {pz2} stone",
            # Clear air above for building and walking
            f"fill {px1} {oy} {pz1} {px2} {clear_top} {pz2} air",
        ]

        cmds = [
            "gamemode survival @s",
            # Survival mode but no damage / hunger
            "effect give @s minecraft:saturation 1000000 255 true",
            "effect give @s minecraft:resistance 1000000 4 true",
            "effect give @s minecraft:fire_resistance 1000000 1 true",
        ]
        cmds = platform_cmds + cmds

        # Give agent blocks based on blueprint
        seen = set()
        for b in self.blueprint.blocks:
            if b.block_name not in seen:
                cmds.append(f"give @s {b.block_name} 64"); seen.add(b.block_name)
        cmds += ["give @s minecraft:dirt 64", "give @s minecraft:scaffolding 64"]

        self._send_commands(cmds)
        noop = _noop_action()
        obs_raw,_,_,_,_ = self.env.step(noop)

        # ── Spawn agent facing the first target ──────────────────
        # Place agent 2 blocks in front of target, facing it.
        # This gives the agent an immediate chance to place a block
        # instead of wandering randomly for hundreds of steps.
        self._retry_cmds = []   # commands to re-run on field reset after success
        self._retry_tp_cmd = None
        if self.blueprint and self.blueprint.build_order:
            b0 = self.blueprint.build_order[0]
            tx, ty, tz = ox+b0.x, oy+b0.y, oz+b0.z
            # Stand 3 blocks away on the +Z side, facing -Z toward target
            sx, sz = tx, tz + 3
            # yaw=180 = facing -Z (north), pitch=10 = aimed at block A top face
            # (block A is at ty-1; from 3 blocks away, target pitch ≈ atan2(0.62,3) ≈ 11.7°)
            self._retry_tp_cmd = f"tp @s {sx}.5 {ty} {sz}.5 180 10"
            # Save platform + give for retry on success (skip gamemode/effects, already active)
            # platform_cmds rebuild the ground; cmds[6+] = give items
            self._retry_cmds = platform_cmds + cmds[6:]
            self._send_commands([self._retry_tp_cmd])
            # Select hotbar slot 1 (holds the block to place)
            hotbar_action = _noop_action()
            hotbar_action['hotbar.1'] = 1
            obs_raw,_,_,_,_ = self.env.step(hotbar_action)
            obs_raw,_,_,_,_ = self.env.step(noop)

        self._pre_step_parsed = _parse_full_obs(obs_raw)
        obs = self._build_obs(obs_raw)

        if self.debug_visual:
            p2 = _parse_full_obs(obs_raw)
            print(f"\n{'='*60}")
            print(f"  RESET — Stage {self.config.curriculum_stage}")
            print(f"  Spawned: ({p2['x']:.1f}, {p2['y']:.1f}, {p2['z']:.1f})"
                  f"  yaw={p2['yaw']:.0f} pitch={p2['pitch']:.0f}")
            print(f"  Build origin: ({ox}, {oy}, {oz})")
            for b in self.blueprint.blocks[:5]:
                wp = (ox+b.x,oy+b.y,oz+b.z)
                print(f"    Target: world{wp} = place {b.block_name}")
            print(f"{'='*60}\n")
            self._debug_window(obs_raw, obs, 0.0, BuildAction.NO_OP)

        return obs, info

    def step(self, action):
        self._step_count += 1
        real_action = action
        pre_parsed = self._pre_step_parsed

        # Track active hotbar slot
        if action == BuildAction.HOTBAR_1:
            self._current_hotbar = 1
        elif action == BuildAction.HOTBAR_2:
            self._current_hotbar = 2
        elif action == BuildAction.HOTBAR_3:
            self._current_hotbar = 3


        # ── Place gate: suppress PLACE when aim error exceeds threshold ───────
        _place_gated = False
        if (action in (BuildAction.PLACE_BLOCK, BuildAction.JUMP_AND_PLACE)
                and pre_parsed is not None
                and self.config.place_aim_gate_threshold > 0):
            t = self._next_target_world()
            if t is not None:
                ye, pe = self._aim_error(pre_parsed, t)
                if math.sqrt(ye**2 + pe**2) > self.config.place_aim_gate_threshold:
                    real_action = BuildAction.NO_OP
                    _place_gated = True

        obs_raw,_,terminated,truncated,info = self.env.step(build_action_to_v2(real_action))
        post_parsed = _parse_full_obs(obs_raw)
        self._pre_step_parsed = post_parsed

        reward = self._reward_build(post_parsed, pre_parsed, real_action, info)
        if _place_gated:
            reward += self.config.reward_place_bad_aim
            self._rc['p_place_gate'] += self.config.reward_place_bad_aim


        if self._step_count >= self.config.max_timesteps: truncated = True
        if self._is_complete():
            scaf_pen = len(self._scaffold_blocks) * self.config.penalty_remaining_scaffold
            reward += self.config.reward_structure_complete + scaf_pen
            self._rc['r_complete'] += self.config.reward_structure_complete
            self._rc['p_scaffold']  += scaf_pen
            info["structure_complete"] = True
            # Reset field and respawn — episode continues until time limit
            obs_raw = self._reset_for_retry()
        if post_parsed['is_dead']:
            reward += self.config.penalty_death
            self._rc['p_death'] += self.config.penalty_death
            terminated = True
        if (terminated or truncated) and not info.get("structure_complete"):
            scaf_pen = len(self._scaffold_blocks) * self.config.penalty_remaining_scaffold
            reward += scaf_pen
            self._rc['p_scaffold'] += scaf_pen

        self._total_reward += reward
        obs = self._build_obs(obs_raw)
        info.update({
            "blocks_placed": len(self._correctly_placed),
            "total_blocks": len(self.blueprint.blocks) if self.blueprint else 0,
            "completion_pct": self._completion(),
            "scaffold_blocks": len(self._scaffold_blocks),
            "total_reward": self._total_reward,
            "step_count": self._step_count,
            "agent_pos": obs["agent_pos"],
        })
        if terminated or truncated:
            info["reward_components"] = dict(self._rc)
        if self.debug_visual:
            self._debug_window(obs_raw, obs, reward, real_action)
        return obs, reward, terminated, truncated, info

    # ── Field reset after successful placement (episode continues) ────
    def _reset_for_retry(self):
        """Clear placed blocks, respawn agent at start pos. Episode keeps going."""
        noop = _noop_action()
        if self._retry_cmds:
            self._send_commands(self._retry_cmds)
        if self._retry_tp_cmd:
            self._send_commands([self._retry_tp_cmd])
        obs_raw,_,_,_,_ = self.env.step(noop)
        hotbar_action = _noop_action(); hotbar_action['hotbar.1'] = 1
        obs_raw,_,_,_,_ = self.env.step(hotbar_action)
        obs_raw,_,_,_,_ = self.env.step(noop)
        # Reset tracking state (keep _step_count — episode continues)
        self._world_blocks.clear(); self._scaffold_blocks.clear()
        self._correctly_placed.clear()
        self._prev_agent_pos = None; self._current_hotbar = 1
        self._aim_streak = 0; self._aim_streak_reward = 0.0
        self._was_on_target = False; self._last_info = ""
        self._pre_step_parsed = _parse_full_obs(obs_raw)
        return obs_raw

    # ── Marker hit detection (by POSITION only) ───────────────────
    def _reward_build(self, post_p, pre_p, action, info):
        rc = self._rc  # shorthand to per-episode accumulator

        reward = self.config.penalty_time_step
        rc['p_time'] += self.config.penalty_time_step

        my = max(self.blueprint.size_y,1) if self.blueprint else 1
        t = self._next_target_world()

        if action in (BuildAction.PLACE_BLOCK, BuildAction.JUMP_AND_PLACE) and pre_p:
            pl = self._detect_placement(pre_p)
            r_=pl["result"]; by=pl["blueprint_y"]
            hm=1.0+self.config.height_reward_scale*(by/my)
            if r_=="exact_match":
                v = self.config.reward_correct_place*hm
                reward += v; rc['r_correct'] += v
                cl=self._current_layer()
                if by==cl:
                    reward+=self.config.reward_in_order
                    rc['r_correct'] += self.config.reward_in_order
                elif by>cl:
                    reward+=self.config.penalty_out_of_order
                    rc['p_wrong'] += self.config.penalty_out_of_order
                if self._layer_complete():
                    reward+=self.config.reward_layer_complete
                    rc['r_correct'] += self.config.reward_layer_complete
                self._last_info=f"EXACT at {pl['pos']}"
            elif r_=="wrong_type":
                v = self.config.reward_partial_place*hm
                reward += v; rc['r_correct'] += v * 0.3  # partial credit
            elif r_ in ("scaffold","wrong"):
                v = self.config.reward_wrong_place * (0.3 if r_=="scaffold" else 1.0)
                reward += v; rc['p_wrong'] += v
                if t and pl['pos']:
                    md=(abs(pl['pos'][0]-t[0])+abs(pl['pos'][1]-t[1])+abs(pl['pos'][2]-t[2]))
                    if md<=1:
                        reward+=self.config.reward_near_miss
                        rc['r_near_miss'] += self.config.reward_near_miss
                        self._last_info=f"near-miss dist={md}"
            elif r_=="none": self._last_info="place: miss"
        elif action==BuildAction.BREAK_BLOCK:
            br=self._detect_break(post_p)
            if br=="scaffold_cleanup":
                reward+=self.config.reward_scaffold_cleanup
                rc['r_scaffold'] += self.config.reward_scaffold_cleanup
            elif br=="correct_destroyed":
                v = self.config.reward_wrong_place*2
                reward += v; rc['p_wrong'] += v
            if self.config.curriculum_stage <= 0 and pre_p and pre_p['raycast_type']=='BLOCK':
                ox2,oy2,oz2 = self.config.build_origin
                rbx = pre_p['raycast_block_x'] - ox2
                rby = pre_p['raycast_block_y'] - oy2
                rbz = pre_p['raycast_block_z'] - oz2
                if (self.blueprint and self.blueprint.grid is not None
                        and 0 <= rbx < self.blueprint.size_x
                        and 0 <= rby+1 < self.blueprint.size_y
                        and 0 <= rbz < self.blueprint.size_z
                        and self.blueprint.grid[rbx, rby+1, rbz] > 0):
                    reward += self.config.reward_wrong_place
                    rc['p_wrong'] += self.config.reward_wrong_place

        ax,ay,az = post_p['x'],post_p['y'],post_p['z']

        # Angle reward: flat bonus when looking roughly toward any target block
        if t:
            ye, pe = self._aim_error(post_p, t)
            td = math.sqrt(ye**2 + pe**2)
            thr = self.config.reward_aim_angle_threshold
            if td < thr:
                v = self.config.reward_aim_angle  # flat reward within threshold
                reward += v; rc['r_aim_angle'] += v

        self._prev_agent_pos=(ax,ay,az)
        return reward

    # ── detection ─────────────────────────────────────────────────
    def _detect_placement(self, parsed):
        no={"result":"none","pos":None,"blueprint_y":0,"expected_block":None}
        pp=_compute_place_position(parsed,parsed['x'],parsed['z'])
        if pp is None: return no
        ox,oy,oz=self.config.build_origin
        bx,by,bz=pp[0]-ox,pp[1]-oy,pp[2]-oz
        if self.debug_visual:
            print(f"  [place] hit=({parsed['raycast_block_x']},{parsed['raycast_block_y']},"
                  f"{parsed['raycast_block_z']}) → pos={pp} bp=({bx},{by},{bz})")

        # ── Blueprint-based placement ───────────────────
        if self.blueprint and self.blueprint.grid is not None:
            if (0<=bx<self.blueprint.size_x and 0<=by<self.blueprint.size_y
                    and 0<=bz<self.blueprint.size_z):
                eid=self.blueprint.grid[bx,by,bz]
                if eid>0:
                    if pp in self._correctly_placed:
                        return {"result":"already_placed","pos":pp,"blueprint_y":by,"expected_block":None}
                    bb=self.blueprint.pos_to_block.get((bx,by,bz))
                    self._correctly_placed.add(pp); self._world_blocks[pp]=eid
                    if self.debug_visual: print(f"    *** EXACT MATCH! ***")
                    return {"result":"exact_match","pos":pp,"blueprint_y":by,
                            "expected_block":bb.block_name if bb else "?"}
                else:
                    self._world_blocks[pp]=1; self._scaffold_blocks.add(pp)
                    return {"result":"wrong","pos":pp,"blueprint_y":by,"expected_block":None}
            else:
                self._world_blocks[pp]=1; self._scaffold_blocks.add(pp)
                return {"result":"scaffold","pos":pp,"blueprint_y":0,"expected_block":None}
        return no

    def _detect_break(self, p):
        if p['raycast_type']!='BLOCK': return "none"
        pos=(p['raycast_block_x'],p['raycast_block_y'],p['raycast_block_z'])
        if pos in self._scaffold_blocks:
            self._scaffold_blocks.discard(pos); self._world_blocks.pop(pos,None); return "scaffold_cleanup"
        if pos in self._correctly_placed:
            self._correctly_placed.discard(pos)
            self._world_blocks.pop(pos,None); return "correct_destroyed"
        self._world_blocks.pop(pos,None); return "other"

    # ── observation ───────────────────────────────────────────────
    def _build_obs(self, obs_raw):
        p=_parse_full_obs(obs_raw)
        ap=np.array([p['x'],p['y'],p['z'],p['yaw'],p['pitch'],p['health']],np.float32)
        r=self.config.local_obs_radius; s=self.obs_size
        lg=np.zeros((s,s,s),np.uint8); dg=np.zeros((s,s,s),np.int8)
        ax,ay,az=int(round(p['x'])),int(round(p['y'])),int(round(p['z']))
        ox,oy,oz=self.config.build_origin

        if self.blueprint and self.blueprint.grid is not None:
            # Local grid: iterate only over placed blocks (sparse — far fewer than 1331)
            for (wx,wy,wz),wid in self._world_blocks.items():
                gx=wx-ax+r; gy=wy-ay+r; gz=wz-az+r
                if 0<=gx<s and 0<=gy<s and 0<=gz<s:
                    lg[gx,gy,gz]=min(wid,255)

            # Vectorised blueprint (target) grid lookup
            off=np.arange(-r,r+1,dtype=np.int32)
            bx_all=(ax+off[:,None,None])-ox   # (s,1,1) broadcasts to (s,s,s)
            by_all=(ay+off[None,:,None])-oy
            bz_all=(az+off[None,None,:])-oz
            valid=((bx_all>=0)&(bx_all<self.blueprint.size_x)&
                   (by_all>=0)&(by_all<self.blueprint.size_y)&
                   (bz_all>=0)&(bz_all<self.blueprint.size_z))
            bx_c=np.clip(bx_all,0,self.blueprint.size_x-1)
            by_c=np.clip(by_all,0,self.blueprint.size_y-1)
            bz_c=np.clip(bz_all,0,self.blueprint.size_z-1)
            tid_all=self.blueprint.grid[bx_c,by_c,bz_c].astype(np.int32)
            tid_all[~valid]=0
            tg=np.clip(tid_all,0,255).astype(np.uint8)

            # Diff grid: vectorised boolean ops
            tg_pos=tg>0; lg_pos=lg>0
            dg[tg_pos&~lg_pos]=1
            dg[~tg_pos&lg_pos]=-1
        else:
            tg=np.zeros((s,s,s),np.uint8)

        total=len(self.blueprint.blocks) if self.blueprint else 1
        placed = len(self._correctly_placed)
        prog=np.array([placed/max(total,1),self._current_layer(),
                        len(self.blueprint.layer_block_counts) if self.blueprint else 1,
                        total-placed],np.float32)

        # ② Raycast grid: mark the block the agent is looking at
        rg=np.zeros((s,s,s),np.float32)
        if p['raycast_type']=='BLOCK':
            rx,ry,rz=p['raycast_block_x'],p['raycast_block_y'],p['raycast_block_z']
            gx,gy,gz=rx-ax+r,ry-ay+r,rz-az+r
            if 0<=gx<s and 0<=gy<s and 0<=gz<s:
                rg[gx,gy,gz]=1.0

        # ③ next_block_id (kept for hotbar selection in later stages)
        t=self._next_target_world()
        if t is not None:
            bx_t, by_t, bz_t = t[0]-ox, t[1]-oy, t[2]-oz
            block_id = 0
            if (self.blueprint and self.blueprint.grid is not None
                    and 0<=bx_t<self.blueprint.size_x
                    and 0<=by_t<self.blueprint.size_y
                    and 0<=bz_t<self.blueprint.size_z):
                block_id = int(self.blueprint.grid[bx_t, by_t, bz_t])
            next_block_id = np.array([block_id / 255.0], np.float32)
        else:
            next_block_id=np.zeros(1,np.float32)

        res={"agent_pos":ap,"local_grid":lg,"target_grid":tg,"diff_grid":dg,
             "raycast_grid":rg,"progress":prog,
             "next_block_id":next_block_id,
             "structure_info":self._structure_info}
        if self.config.use_visual_obs and p['image'] is not None:
            img=p['image']
            if img.shape[:2]!=(self.config.image_height,self.config.image_width):
                img=cv2.resize(img,(self.config.image_width,self.config.image_height))
            res["image"]=img
        return res

    # ── aim error helper ──────────────────────────────────────────
    def _aim_error(self, parsed, t):
        """Signed (yaw_err, pitch_err) in degrees toward the top face of the block below target."""
        ax, ay = parsed['x'], parsed['y']
        tdx = (t[0] + 0.5) - ax
        tdz = (t[2] + 0.5) - parsed['z']
        dh = max(math.sqrt(tdx**2 + tdz**2), 0.01)
        t_yaw = math.degrees(math.atan2(-tdx, tdz))
        # Point toward top face of block below target: use t[1] not t[1]+0.5
        t_pitch = math.degrees(math.atan2(-(t[1] - (ay + 1.62)), dh))
        yaw_err = ((parsed['yaw'] - t_yaw + 180) % 360) - 180  # signed
        pitch_err = parsed['pitch'] - t_pitch                   # signed
        return yaw_err, pitch_err

    # ── reward component helpers ──────────────────────────────────
    def _reset_rc(self):
        self._rc = {
            'r_correct':      0.0,  # correct placements (incl. in-order, layer bonus)
            'r_near_miss':    0.0,  # near-miss placement bonus
            'r_scaffold':     0.0,  # scaffold cleanup reward
            'r_aim_angle':    0.0,  # flat per-step reward when looking toward target area
            'r_complete':     0.0,  # structure completion bonus
            'p_time':         0.0,  # time step penalty
            'p_wrong':        0.0,  # wrong placement / out-of-order penalties
            'p_place_gate':   0.0,  # suppressed-place penalty
            'p_scaffold':     0.0,  # remaining scaffold penalty at episode end
            'p_death':        0.0,  # death penalty
        }

    # ── block-A aim helpers ───────────────────────────────────────
    def _is_aiming_at_A(self, parsed, t) -> bool:
        """True if raycast hits the TOP face of block A (the block one below target t)."""
        if parsed['raycast_type'] != 'BLOCK':
            return False
        face = str(parsed.get('raycast_face', '')).upper()
        return (
            parsed['raycast_block_x'] == t[0] and
            parsed['raycast_block_y'] == t[1] - 1 and
            parsed['raycast_block_z'] == t[2] and
            ('UP' in face or 'TOP' in face)
        )

    def is_aiming_at_target_base(self) -> bool:
        """Returns True if the pre-step raycast is hitting block A's top face.
        Used by the trainer to bias PLACE logit for guided exploration."""
        p = self._pre_step_parsed
        t = self._next_target_world()
        if p is None or t is None:
            return False
        return self._is_aiming_at_A(p, t)

    # ── structure info for observation ───────────────────────────
    def _compute_structure_info(self):
        """Compute normalised structure info vector: [type_id, n_blocks, size_x, size_y, size_z]."""
        stage = self.config.curriculum_stage
        type_map = {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.0}
        type_id = type_map.get(stage, 1.0)
        n = len(self.blueprint.blocks) if self.blueprint else 1
        sx = self.blueprint.size_x if self.blueprint else 1
        sy = self.blueprint.size_y if self.blueprint else 1
        sz = self.blueprint.size_z if self.blueprint else 1
        return np.array([type_id, n / 100.0, sx / 20.0, sy / 20.0, sz / 20.0], np.float32)

    # ── progress ──────────────────────────────────────────────────
    def _completion(self):
        if not self.blueprint or not self.blueprint.blocks: return 0.0
        return len(self._correctly_placed)/len(self.blueprint.blocks)
    def _is_complete(self):
        return self.blueprint and len(self._correctly_placed)>=len(self.blueprint.blocks)
    def _current_layer(self):
        if not self.blueprint: return 0
        ox,oy,oz=self.config.build_origin
        for yl in sorted(self.blueprint.layer_block_counts):
            for b in self.blueprint.blocks:
                if b.y==yl and (ox+b.x,oy+b.y,oz+b.z) not in self._correctly_placed: return yl
        return max(self.blueprint.layer_block_counts,default=0)
    def _layer_complete(self):
        c=self._current_layer(); ox,oy,oz=self.config.build_origin
        return all((ox+b.x,oy+b.y,oz+b.z) in self._correctly_placed for b in self.blueprint.blocks if b.y==c)
    def _get_first_target_world(self):
        if not self.blueprint or not self.blueprint.build_order: return None
        b=self.blueprint.build_order[0]; ox,oy,oz=self.config.build_origin
        return (ox+b.x,oy+b.y,oz+b.z)
    def _next_target_world(self):
        if not self.blueprint: return None
        ox,oy,oz=self.config.build_origin
        for b in self.blueprint.build_order:
            wp=(ox+b.x,oy+b.y,oz+b.z)
            if wp not in self._correctly_placed: return wp
        return None

    def _next_target_rel(self,ax,ay,az):
        t=self._next_target_world()
        if t is None: return np.zeros(3,np.float32)
        return np.array([t[0]-ax,t[1]-ay,t[2]-az],np.float32)
    def _send_commands(self, cmds):
        try: self.get_wrapper_attr("add_commands")(cmds); return
        except: pass
        try:
            for c in cmds: self.env.add_command(c)
        except: pass

    # ── 3-D world → 2-D screen projection ────────────────────────
    @staticmethod
    def _world_to_screen(ax, ay, az, yaw, pitch, tx, ty, tz, sw, sh, fov_y=70.0):
        """Returns (sx, sy) pixel coords, or None if behind camera."""
        yr, pr = math.radians(yaw), math.radians(pitch)
        # forward / right / up in Minecraft convention
        fwd = (-math.sin(yr)*math.cos(pr), -math.sin(pr),  math.cos(yr)*math.cos(pr))
        rgt = ( math.cos(yr),               0.0,            math.sin(yr))
        uup = ( math.sin(yr)*math.sin(pr),  math.cos(pr), -math.cos(yr)*math.sin(pr))
        dx, dy, dz = tx - ax, ty - (ay + 1.62), tz - az
        cz = dx*fwd[0] + dy*fwd[1] + dz*fwd[2]
        if cz <= 0.1:
            return None
        cx = dx*rgt[0] + dy*rgt[1] + dz*rgt[2]
        cy = dx*uup[0] + dy*uup[1] + dz*uup[2]
        f  = (sh / 2.0) / math.tan(math.radians(fov_y) / 2.0)
        sx = int(cx / cz * f + sw / 2.0)
        sy = int(-cy / cz * f + sh / 2.0)
        return sx, sy

    # ── debug window ──────────────────────────────────────────────
    def _debug_window(self, obs_raw, obs, reward, action):
        if not HAS_CV2: return
        p=_parse_full_obs(obs_raw)
        img=p.get('image')
        if img is None or not isinstance(img,np.ndarray):
            img=np.zeros((256,256,3),np.uint8)
        dw,dh=540,400
        if img.shape[0]!=dh or img.shape[1]!=dw:
            img=cv2.resize(img,(dw,dh),interpolation=cv2.INTER_NEAREST)
        if len(img.shape)==3 and img.shape[2]==3:
            img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        target = self._next_target_world()

        # Draw red diamond overlay on image for target position
        if target is not None:
            sp = self._world_to_screen(
                p['x'], p['y'], p['z'], p['yaw'], p['pitch'],
                target[0]+0.5, target[1]+0.5, target[2]+0.5,
                dw, dh,
            )
            if sp is not None:
                sx, sy = sp
                r = max(6, int(200 / max(1, abs(sx - dw//2) + abs(sy - dh//2) + 1)))
                diamond = np.array([
                    [sx,     sy - r],
                    [sx + r, sy    ],
                    [sx,     sy + r],
                    [sx - r, sy    ],
                ], dtype=np.int32)
                cv2.polylines(img, [diamond], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.line(img, (sx - r//2, sy), (sx + r//2, sy), (0, 0, 255), 1)
                cv2.line(img, (sx, sy - r//2), (sx, sy + r//2), (0, 0, 255), 1)

        # Panel
        pw=340; pan=np.full((dh,pw,3),(30,30,30),np.uint8)
        aname=BuildAction(action).name if 0<=action<len(BuildAction) else "?"
        pl=len(self._correctly_placed); tot=len(self.blueprint.blocks) if self.blueprint else 0

        lines=[
            (f"Stage {self.config.curriculum_stage}",(255,200,0)),
            (f"Step:{self._step_count}  Act:{aname}",(200,200,0)),
            (f"",None),
            (f"Completion: {self._completion():.0%} ({pl}/{tot})",(0,255,0) if pl>0 else None),
        ]
        lines+=[
            (f"R:{reward:+.3f} Total:{self._total_reward:+.1f}",None),
            (f"",None),
            (f"Pos:({p['x']:.1f},{p['y']:.1f},{p['z']:.1f}) P:{p['pitch']:.0f} Y:{p['yaw']:.0f}",None),
            (f"Ray:{p['raycast_type']}",None),
        ]
        if p['raycast_type']=='BLOCK':
            hit=(p['raycast_block_x'],p['raycast_block_y'],p['raycast_block_z'])
            lines.append((f" hit={hit} {p['raycast_block_name'][:15]}", None))
        if target:
            lines+=[(f"",None),(f"Target:{target}",(0,255,255))]
            yd,pd,td=_angle_to_target(p['x'],p['y'],p['z'],p['yaw'],p['pitch'],
                                       target[0]+.5,target[1]+.5,target[2]+.5)
            lines.append((f"Angle:{td:.1f} (y:{yd:.0f} p:{pd:.0f})",
                          (0,255,0) if td<15 else None))
            ray_hit = obs["raycast_grid"].sum() > 0
            lines.append((f"Raycast hit: {ray_hit}",(0,200,200)))
        if self._last_info:
            col=(0,255,0) if 'HIT' in self._last_info or 'EXACT' in self._last_info else (180,180,180)
            lines.append((self._last_info[:42],col))

        y=16
        for txt,col in lines:
            cv2.putText(pan,txt,(5,y),cv2.FONT_HERSHEY_SIMPLEX,0.36,col or(180,180,180),1,cv2.LINE_AA)
            y+=16
        cv2.imshow("CraftGround Debug",np.hstack([img,pan]))
        cv2.waitKey(1)

    def close(self):
        if self.debug_visual and HAS_CV2: cv2.destroyAllWindows()
        super().close()


# ══════════════════════════════════════════════════════════════════
def make_building_env(
    blueprint, port=8023, build_origin=(0,-60,0), curriculum_stage=0,
    image_size=64, max_timesteps=2000, seed=12345, debug_visual=False,
    blueprint_generator=None, structure_name="", place_aim_gate=25.0,
):
    from craftground import CraftGroundEnvironment, InitialEnvironmentConfig
    from craftground.initial_environment_config import GameMode, Difficulty, WorldType
    try:    from craftground import ActionSpaceVersion
    except: from craftground.environment import ActionSpaceVersion

    ox, oy, oz = build_origin
    cmds = [
        # Build a small stone platform at spawn so agent doesn't fall
        f"fill {ox-3} {oy-3} {oz-3} {ox+3} {oy-1} {oz+3} stone",
        f"fill {ox-3} {oy} {oz-3} {ox+3} {oy+10} {oz+3} air",
        "gamemode survival @s",
        "gamerule doDaylightCycle false","gamerule doWeatherCycle false",
        "gamerule doMobSpawning false","gamerule doFireTick false",
        "gamerule randomTickSpeed 0","gamerule sendCommandFeedback false",
        "gamerule keepInventory true",
        "time set day","difficulty peaceful",
        f"tp @s {ox} {oy} {oz}",   # fixed spawn — world loads at this position
    ]
    vw,vh,hud = (640,360,False) if debug_visual else (image_size,image_size,True)

    ic = InitialEnvironmentConfig(
        image_width=vw,image_height=vh,
        gamemode=GameMode.SURVIVAL,difficulty=Difficulty.PEACEFUL,
        world_type=WorldType.DEFAULT,
        seed=str(seed) if seed is not None else "",
        initial_extra_commands=cmds,
        hud_hidden=hud,render_distance=6,simulation_distance=6,
        request_raycast=True,no_fov_effect=True,
    )
    env = CraftGroundEnvironment(
        initial_env=ic,port=port,
        action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        render_action=True,
    )
    # FastResetWrapper intentionally not used — full world reload per episode
    # ensures placed blocks are cleared and terrain is restored naturally.

    cfg = BuildingConfig(
        blueprint=blueprint,build_origin=build_origin,
        image_width=image_size,image_height=image_size,
        max_timesteps=max_timesteps,curriculum_stage=curriculum_stage,
        structure_name=structure_name,
        blueprint_generator=blueprint_generator,
        place_aim_gate_threshold=place_aim_gate,
    )
    return HouseBuildingWrapper(env, cfg, debug_visual=debug_visual)
