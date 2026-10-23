"""Record three preselected successful ACT confirmation rollouts, without takeover."""
from pathlib import Path
import json, subprocess, hashlib
import cv2
import numpy as np
import torch
from scripts.eval import run_episode, XML_PATH_MULTICUBE
from hw3.sim_env import SO100MulticubeSimEnv, CUBE_COLORS
from hw3.eval_utils import load_checkpoint, check_success

ROOT=Path(__file__).resolve().parents[1]
CKPT=ROOT/'checkpoints/multi_cube/act_scene_dagger36_mse_kl01/epoch090_ee_xyz_act_scene_dagger36ep_width256.pt'
device=torch.device('cuda');assert torch.cuda.is_available()
model,norm,h,sk,ak=load_checkpoint(CKPT,device)
assert h==16 and ak==['action_ee_xyz','action_gripper']
env=SO100MulticubeSimEnv(xml_path=XML_PATH_MULTICUBE,render_w=640,render_h=480,use_mocap=True,goal_cube='red',shuffle_cubes=True,seed=4242)
orig_step=env.step
selected={2:'green',6:'blue',7:'red'}
metadata=[];episode=0;writer=None;step=0;last=None
def frame():
    img=np.concatenate([env.render('angle'),env.render('top')],axis=1)
    header=np.zeros((88,img.shape[1],3),dtype=np.uint8)
    color=env.goal_cube.upper()
    cv2.putText(header,f'ACT | Target: {color} | Epoch 90 | Width 256 | Horizon 16',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.73,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(header,f'Autonomous rollout | Seed 4242 | Episode {episode} | Step {step}',(18,66),cv2.FONT_HERSHEY_SIMPLEX,.62,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(header,'3x playback',(1090,66),cv2.FONT_HERSHEY_SIMPLEX,.62,(60,230,250),2,cv2.LINE_AA)
    return np.concatenate([header,img],axis=0)
def recording_step():
    global step,last
    obs=orig_step();step+=1
    if writer is not None:
        last=frame();writer.write(last)
    return obs
env.step=recording_step
try:
    for episode in range(1,8):
        color=CUBE_COLORS[(episode-1)%3];env.set_goal(color);step=0;writer=None
        temp=Path(f'/tmp/act_{color}_episode{episode}.mp4')
        if episode in selected:
            assert color==selected[episode]
            writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),30,(1280,568));assert writer.isOpened()
        success,aborted,wrong=run_episode(env,model,norm,sk,ak,device,800,0,episode-1,True,True,16)
        print(f'Episode {episode} {color}: success={success}, steps={step}',flush=True)
        if writer is not None:
            assert success and not aborted and wrong is None, 'Selected confirmation success did not reproduce'
            # Hold the final frame; do not apply extra actions or alter the evaluation.
            cv2.putText(last,'TARGET IN BIN',(20,120),cv2.FONT_HERSHEY_SIMPLEX,.8,(80,240,80),2,cv2.LINE_AA)
            for _ in range(20):writer.write(last)
            writer.release();writer=None
            dest=ROOT/f'media/act_{color}_autonomous_3x.mp4'
            assert not dest.exists(),dest
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(temp),'-c:v','libx264','-crf','21','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
            cv2.imwrite(str(ROOT/f'media/act_{color}_preview.jpg'),last)
            metadata.append(dict(color=color,checkpoint=str(CKPT.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),seed=4242,episode=episode,steps=step,success=success,wrong_cube_in_bin=wrong,device=str(device),execution_horizon=h,source_fps=10,encoded_fps=30,playback_speed=3,final_frame_hold_source_seconds=2,autonomous=True,video=dest.name,selection='First successful episode of this color in the saved seed-4242 confirmation log; not an aggregate estimate.'))
finally:
    if writer is not None:writer.release()
    env.renderer.close()
(ROOT/'media/act_videos.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Recorded all three autonomous ACT videos.',flush=True)
