import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import io, base64
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nexus-Steg", layout="wide")
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0d0f14; color:#e8eaf0; }
  [data-testid="stSidebar"]          { background:#13161e; }
  .block-container                   { padding-top:1.5rem; }
  h1,h2,h3                           { color:#a8d8ff; }
  [data-testid="stImage"] img        { object-fit:contain; }
  .stButton > button {
      background:linear-gradient(135deg,#1e6fff,#7b2fff);
      color:#fff; border:none; border-radius:8px;
      padding:0.45rem 1.2rem; font-weight:600;
  }
  .stButton > button:hover { opacity:0.85; }
  .metric-card {
      background:#1a1d26; border:1px solid #2a2d3a;
      border-radius:10px; padding:1rem; text-align:center;
  }
</style>
""", unsafe_allow_html=True)

st.title("Nexus-Steg · HiNet Image Steganography")
st.caption("Invertible Neural Network — hide a secret image inside a cover, recover it after attacks.")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "hinet_final.pth")

# ═══════════════════════════════════════════════════════════════
#  DWT / IWT
# ═══════════════════════════════════════════════════════════════
def dwt_init(x):
    x01=x[:,:,0::2,:]/2; x02=x[:,:,1::2,:]/2
    x1=x01[:,:,:,0::2]; x2=x02[:,:,:,0::2]
    x3=x01[:,:,:,1::2]; x4=x02[:,:,:,1::2]
    return torch.cat((x1+x2+x3+x4,-x1-x2+x3+x4,-x1+x2-x3+x4,x1-x2-x3+x4),1)

def iwt_init(x):
    r=2; ib,ic,ih,iw=x.size(); oc=ic//(r**2)
    x1=x[:,0:oc,:,:]/2; x2=x[:,oc:oc*2,:,:]/2
    x3=x[:,oc*2:oc*3,:,:]/2; x4=x[:,oc*3:oc*4,:,:]/2
    h=torch.zeros(ib,oc,r*ih,r*iw,device=x.device,dtype=x.dtype)
    h[:,:,0::2,0::2]=x1-x2-x3+x4; h[:,:,1::2,0::2]=x1-x2+x3-x4
    h[:,:,0::2,1::2]=x1+x2-x3-x4; h[:,:,1::2,1::2]=x1+x2+x3+x4
    return h

class DWT(nn.Module):
    def forward(self,x): return dwt_init(x)
class IWT(nn.Module):
    def forward(self,x): return iwt_init(x)

# ═══════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════
CLAMP=2.0

class ResidualDenseBlock_out(nn.Module):
    def __init__(self,in_ch,out_ch,bias=True):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,    32,    3,1,1,bias=bias)
        self.conv2=nn.Conv2d(in_ch+32, 32,    3,1,1,bias=bias)
        self.conv3=nn.Conv2d(in_ch+64, 32,    3,1,1,bias=bias)
        self.conv4=nn.Conv2d(in_ch+96, 32,    3,1,1,bias=bias)
        self.conv5=nn.Conv2d(in_ch+128,out_ch,3,1,1,bias=bias)
        self.lrelu=nn.LeakyReLU(inplace=True)
        nn.init.constant_(self.conv5.weight,0.)
        nn.init.constant_(self.conv5.bias,0.)
    def forward(self,x):
        x1=self.lrelu(self.conv1(x))
        x2=self.lrelu(self.conv2(torch.cat((x,x1),1)))
        x3=self.lrelu(self.conv3(torch.cat((x,x1,x2),1)))
        x4=self.lrelu(self.conv4(torch.cat((x,x1,x2,x3),1)))
        return self.conv5(torch.cat((x,x1,x2,x3,x4),1))

class INV_block(nn.Module):
    def __init__(self,in_1=3,in_2=3):
        super().__init__()
        self.split_len1=in_1*4; self.split_len2=in_2*4; self.clamp=CLAMP
        self.r=ResidualDenseBlock_out(self.split_len1,self.split_len2)
        self.y=ResidualDenseBlock_out(self.split_len1,self.split_len2)
        self.f=ResidualDenseBlock_out(self.split_len2,self.split_len1)
    def e(self,s): return torch.exp(self.clamp*2*(torch.sigmoid(s)-0.5))
    def forward(self,x,rev=False):
        x1=x.narrow(1,0,self.split_len1); x2=x.narrow(1,self.split_len1,self.split_len2)
        if not rev:
            t2=self.f(x2); y1=x1+t2; s1,t1=self.r(y1),self.y(y1); y2=self.e(s1)*x2+t1
        else:
            s1,t1=self.r(x1),self.y(x1); y2=(x2-t1)/self.e(s1); t2=self.f(y2); y1=x1-t2
        return torch.cat((y1,y2),1)

class Hinet(nn.Module):
    def __init__(self): super().__init__(); self.blocks=nn.ModuleList([INV_block() for _ in range(16)])
    def forward(self,x,rev=False):
        if not rev:
            for b in self.blocks: x=b(x)
        else:
            for b in reversed(self.blocks): x=b(x,rev=True)
        return x

class HiNetModel(nn.Module):
    def __init__(self): super().__init__(); self.model=Hinet()
    def forward(self,x,rev=False): return self.model(x,rev=rev)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
IMG_SIZE=256; CHANNELS_IN=3
transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor()])
dwt=DWT(); iwt=IWT()

def psnr(a,b):
    mse=np.mean((a.astype(float)-b.astype(float))**2)
    if mse==0: return 60.0
    return 20*np.log10(255.0/np.sqrt(mse))

def t2pil(t):
    arr=t.squeeze(0).clamp(0,1).permute(1,2,0).detach().numpy()
    return Image.fromarray((arr*255).astype(np.uint8))

def pil2t(img):
    return torch.from_numpy(np.array(img).astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0)

def pil_to_b64(img,fmt="PNG"):
    buf=io.BytesIO(); img.save(buf,format=fmt)
    return "data:image/"+fmt.lower()+";base64,"+base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Checkpoint not found:\n{MODEL_PATH}"
    try:
        net=HiNetModel()
        ckpt=torch.load(MODEL_PATH,map_location="cpu",weights_only=False)
        sd=ckpt["net"]; epoch=ckpt.get("epoch","?")
        net.load_state_dict(sd,strict=True); net.eval()
        return net, f"hinet_final.pth loaded (epoch {epoch}) — strict=True match"
    except Exception as e:
        try:
            net=HiNetModel()
            ckpt=torch.load(MODEL_PATH,map_location="cpu",weights_only=False)
            sd=ckpt.get("net",ckpt); sd={k.replace("module.",""):v for k,v in sd.items()}
            missing,unexpected=net.load_state_dict(sd,strict=False); net.eval()
            return net, f"Loaded strict=False — missing:{len(missing)} unexpected:{len(unexpected)}"
        except Exception as e2:
            return None, f"Load failed:\n{e}\n{e2}"

# ─────────────────────────────────────────────
# run_hide / run_reveal
# ─────────────────────────────────────────────
def run_hide(net,cover,secret):
    c=transform(cover).unsqueeze(0); s=transform(secret).unsqueeze(0)
    inp=torch.cat((dwt(c),dwt(s)),dim=1)
    with torch.no_grad(): out=net(inp)
    out_steg=out.narrow(1,0,4*CHANNELS_IN)
    out_z=out.narrow(1,4*CHANNELS_IN,out.shape[1]-4*CHANNELS_IN)
    return t2pil(iwt(out_steg).clamp(0,1)), out_steg.clone(), out_z.clone()

def run_reveal(net,attacked_t,out_steg,out_z,is_clean):
    if is_clean:
        inp=torch.cat((out_steg,out_z),dim=1)
    else:
        steg_wav=dwt(attacked_t)
        z_gauss=torch.randn_like(steg_wav)
        inp=torch.cat((steg_wav,z_gauss),dim=1)
    with torch.no_grad(): out=net(inp,rev=True)
    secret_wav=out.narrow(1,4*CHANNELS_IN,out.shape[1]-4*CHANNELS_IN)
    return t2pil(iwt(secret_wav).clamp(0,1))

# ─────────────────────────────────────────────
# Attacks
# ─────────────────────────────────────────────
def _jpeg(t,q):
    buf=io.BytesIO(); t2pil(t).save(buf,format="JPEG",quality=q); buf.seek(0)
    return pil2t(Image.open(buf).convert("RGB"))

def atk_clean(t):    return t.clone()
def atk_jpeg90(t):   return _jpeg(t,90)
def atk_jpeg50(t):   return _jpeg(t,50)
def atk_blur(t):     return pil2t(t2pil(t).filter(ImageFilter.GaussianBlur(radius=2)))
def atk_noise(t):    return (t+torch.randn_like(t)*0.03).clamp(0,1)
def atk_resize50(t):
    _,_,h,w=t.shape
    d=F.interpolate(t,size=(h//2,w//2),mode="bilinear",align_corners=False)
    return F.interpolate(d,size=(h,w),mode="bilinear",align_corners=False)
def atk_resize75(t):
    _,_,h,w=t.shape; nh,nw=max(1,int(h*.75)),max(1,int(w*.75))
    d=F.interpolate(t,size=(nh,nw),mode="bilinear",align_corners=False)
    return F.interpolate(d,size=(h,w),mode="bilinear",align_corners=False)
def atk_social(t):   return _jpeg(atk_resize75(t),70)

ATTACKS={
    "Clean (no attack)":             (atk_clean,    True,  "Lossless — uses real z from hide. Maximum PSNR."),
    "JPEG quality=90 (light)":       (atk_jpeg90,   False, "Light JPEG compression (quality=90)."),
    "JPEG quality=50 (medium)":      (atk_jpeg50,   False, "Medium JPEG compression (quality=50)."),
    "Gaussian blur (radius=2)":      (atk_blur,     False, "PIL GaussianBlur radius=2."),
    "Gaussian noise (std=0.03)":     (atk_noise,    False, "Additive white Gaussian noise σ=0.03."),
    "Resize ×0.50 (down+up)":        (atk_resize50, False, "Bilinear downsample 50% then upsample."),
    "Resize ×0.75 (down+up)":        (atk_resize75, False, "Bilinear downsample 75% then upsample."),
    "Social media (resize+JPEG 70)": (atk_social,   False, "Resize×0.75 → JPEG quality=70."),
}

# ─────────────────────────────────────────────
# Pipeline HTML builder
# ─────────────────────────────────────────────
def build_pipeline_html(secret_b64, cover_b64, auto_mode="idle",
                        stego_b64=None, recovered_b64=None):
    stego_src = stego_b64     if stego_b64     else cover_b64
    rec_src   = recovered_b64 if recovered_b64 else secret_b64

    # Forward pass: hide entire controls bar (auto-plays once, no replay needed)
    # Reveal pass:  show only "Replay reveal" + Reset + Speed
    ctrls_style      = "display:none" if auto_mode == "fwd" else ""
    rev_btn_disabled = "" if auto_mode == "rev" else "disabled"

    return f"""<!DOCTYPE html><html><head>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d0f14;font-family:'Courier New',monospace;color:#e8eaf0;overflow-x:hidden}}
#phasebar{{display:flex;align-items:center;gap:5px;padding:7px 12px;background:#13161e;border-bottom:1px solid #1e2130}}
.pd{{flex:1;height:3px;border-radius:2px;background:#1e2130;transition:background .3s}}
.pd.on{{background:#3b82f6}}.pd.ok{{background:#10b981}}
#plbl{{font-size:11px;color:#9ca3af;flex:5;padding-left:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
#cw{{background:#0d0f14;width:100%}}
canvas#C{{display:block;width:100%}}
#ctrls{{display:flex;gap:8px;padding:7px 12px;background:#13161e;border-top:1px solid #1e2130;flex-wrap:wrap;align-items:center}}
.btn{{border:none;border-radius:6px;padding:6px 14px;font-family:inherit;font-size:11px;font-weight:700;cursor:pointer;transition:opacity .15s}}
.btn:disabled{{opacity:.35;cursor:default}}
.btn.p{{background:linear-gradient(135deg,#1d4ed8,#7c3aed);color:#fff}}
.btn.p:hover:not(:disabled){{opacity:.85}}
.btn.g{{background:#1e2130;border:1px solid #2a2d3a;color:#9ca3af}}
.btn.g:hover:not(:disabled){{background:#252836}}
#sp{{display:flex;align-items:center;gap:5px;margin-left:auto;font-size:10px;color:#6b7280}}
#sp input{{width:64px}}
</style>
</head><body>
<div id="phasebar">
  <div class="pd" id="d0"></div><div class="pd" id="d1"></div><div class="pd" id="d2"></div>
  <div class="pd" id="d3"></div><div class="pd" id="d4"></div><div class="pd" id="d5"></div>
  <div class="pd" id="d6"></div><div class="pd" id="d7"></div>
  <span id="plbl">Loading...</span>
</div>
<div id="cw"><canvas id="C"></canvas></div>
<div id="ctrls" style="{ctrls_style}">
  <button class="btn p" id="bR" {rev_btn_disabled}>Replay reveal</button>
  <button class="btn g" id="bX">Reset</button>
  <div id="sp"><span>Speed</span><input type="range" id="spR" min="1" max="5" step="1" value="2"><span id="spL">2x</span></div>
</div>
<script>
const C=document.getElementById('C');
const X=C.getContext('2d');
const DPR=Math.min(window.devicePixelRatio||1,2);

const SECRET_SRC="{secret_b64}";
const COVER_SRC="{cover_b64}";
const STEGO_SRC="{stego_src}";
const REC_SRC="{rec_src}";
const AUTO_MODE="{auto_mode}";

let sOff=null,cOff=null,sWav=null,cWav=null,stegoOff=null,recOff=null;
let mode='idle',tick=0,animating=false,raf=null,spd=1;

document.getElementById('spR').oninput=function(){{spd=+this.value;document.getElementById('spL').textContent=spd+'x';}};

const OSZ=160;
function mkOff(img){{const c=document.createElement('canvas');c.width=c.height=OSZ;c.getContext('2d').drawImage(img,0,0,OSZ,OSZ);return c;}}
function mkWav(src,col){{
  const c=document.createElement('canvas');c.width=c.height=OSZ;const x=c.getContext('2d'),h=OSZ/2;
  x.drawImage(src,0,0,h,h);x.fillStyle='rgba(0,0,0,.22)';x.fillRect(0,0,h,h);
  x.globalAlpha=.55;x.drawImage(src,h,0,h,h);x.fillStyle=col+'44';x.fillRect(h,0,h,h);
  x.globalAlpha=.35;x.drawImage(src,0,h,h,h);x.fillStyle=col+'33';x.fillRect(0,h,h,h);
  x.globalAlpha=.18;x.drawImage(src,h,h,h,h);x.fillStyle='#00000055';x.fillRect(h,h,h,h);
  x.globalAlpha=1;x.strokeStyle=col;x.lineWidth=1.6;
  x.beginPath();x.moveTo(h,0);x.lineTo(h,OSZ);x.stroke();
  x.beginPath();x.moveTo(0,h);x.lineTo(OSZ,h);x.stroke();
  ['LL','HL','LH','HH'].forEach((n,i)=>{{x.fillStyle=col;x.font='bold 14px monospace';x.fillText(n,(i%2)*h+4,Math.floor(i/2)*h+16);}});
  return c;
}}
function loadImg(src){{return new Promise(r=>{{const i=new Image();i.onload=()=>r(i);i.src=src;}});}}

async function init(){{
  const [sI,cI,stI,rI]=await Promise.all([loadImg(SECRET_SRC),loadImg(COVER_SRC),loadImg(STEGO_SRC),loadImg(REC_SRC)]);
  sOff=mkOff(sI);cOff=mkOff(cI);sWav=mkWav(sOff,'#3b82f6');cWav=mkWav(cOff,'#ec4899');
  stegoOff=mkOff(stI);recOff=mkOff(rI);
  setPL('Ready');
  if(AUTO_MODE==='fwd') startFwd();
  else if(AUTO_MODE==='rev') startRev();
  else draw();
}}
init();

// ── Layout (verified: zero overlaps) ────────────────────────
const VW=1110, VH=360;
const IW=88,IH=88,BW=24,BH=42,BGAP=7,NB=16;
const BLK_TOTAL=NB*(BW+BGAP)-BGAP; // 489
const img_x=12,dwt_x=118,cat_bx=224,blk_x=286;
const split_x=785,iwt_x=900,out_x=1006;
const sY=42,cY=186;
const midY=Math.round((sY+IH/2+cY+IH/2)/2); // 158
const stegoForkY=sY+IH/2; // 86
const iwtY=sY;             // 42 — bottom=130, cY=186, gap=56 ✓
const zForkY=cY+IH/2;     // 230

function setupCanvas(){{
  const w=C.offsetWidth||700;
  C.width=VW*DPR;C.height=VH*DPR;
  C.style.width=w+'px';C.style.height=Math.round(VH*(w/VW))+'px';
  X.setTransform(DPR,0,0,DPR,0,0);
}}

function imgBox(src,x,y,w,h,a,tlbl,slbl,col,glow){{
  if(a<=0)return;
  X.save();X.globalAlpha=Math.min(a,1);
  if(tlbl){{X.font='600 11px Courier New,monospace';X.fillStyle=col||'#9ca3af';X.textAlign='center';X.textBaseline='alphabetic';X.fillText(tlbl,x+w/2,y-14);}}
  if(slbl){{X.font='400 9px Courier New,monospace';X.fillStyle='#6b7280';X.textAlign='center';X.textBaseline='alphabetic';X.fillText(slbl,x+w/2,y-3);}}
  X.beginPath();X.roundRect(x,y,w,h,6);X.fillStyle='#13161e';X.fill();
  if(src){{X.save();X.beginPath();X.roundRect(x,y,w,h,6);X.clip();X.imageSmoothingEnabled=true;X.imageSmoothingQuality='high';X.drawImage(src,x,y,w,h);X.restore();}}
  X.strokeStyle=glow?col:'#374151';X.lineWidth=glow?2:1;X.stroke();
  X.restore();
}}
function blk(x,y,w,h,lbl,col,a,pulse){{
  if(a<=0)return;
  X.save();X.globalAlpha=Math.min(a,1);
  X.beginPath();X.roundRect(x,y,w,h,5);X.fillStyle='#13161e';X.fill();
  X.strokeStyle=col;X.lineWidth=1.4;X.stroke();
  if(pulse>0){{X.strokeStyle=col;X.lineWidth=1;X.globalAlpha=a*(1-pulse)*.45;X.beginPath();X.arc(x+w/2,y+h/2,h*.9+pulse*8,0,Math.PI*2);X.stroke();}}
  X.globalAlpha=Math.min(a,1);
  X.font='600 10px Courier New,monospace';X.fillStyle=col;X.textAlign='center';X.textBaseline='middle';X.fillText(lbl,x+w/2,y+h/2);
  X.restore();
}}
function arrow(x1,y1,x2,y2,col,a,dash){{
  if(a<=0)return;
  X.save();X.globalAlpha=Math.min(a,1);X.strokeStyle=col;X.lineWidth=1.4;
  if(dash)X.setLineDash([5,4]);
  X.beginPath();X.moveTo(x1,y1);X.lineTo(x2,y2);X.stroke();X.setLineDash([]);
  const ang=Math.atan2(y2-y1,x2-x1);
  X.fillStyle=col;X.beginPath();X.moveTo(x2,y2);
  X.lineTo(x2-8*Math.cos(ang-.4),y2-8*Math.sin(ang-.4));
  X.lineTo(x2-8*Math.cos(ang+.4),y2-8*Math.sin(ang+.4));
  X.closePath();X.fill();X.restore();
}}
function txt(t,x,y,sz,col,al){{
  X.save();X.font=`400 ${{sz||11}}px Courier New,monospace`;
  X.fillStyle=col||'#9ca3af';X.textAlign=al||'center';X.textBaseline='middle';X.fillText(t,x,y);X.restore();
}}
function ease(t){{return t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;}}
function pr(f,s,e){{return Math.max(0,Math.min(1,(f-s)/(e-s)));}}
function setPL(t){{document.getElementById('plbl').textContent=t;}}
function setPD(cur){{for(let i=0;i<8;i++){{const d=document.getElementById('d'+i);d.className='pd'+(i<cur?' ok':i===cur?' on':'');}}}}

function draw(){{
  setupCanvas();X.clearRect(0,0,VW,VH);X.fillStyle='#0d0f14';X.fillRect(0,0,VW,VH);
  if(!sOff){{txt('Loading...',VW/2,VH/2,12,'#4b5563');return;}}
  if(mode==='fwd'||mode==='fwdDone'){{drawFwd(tick);}}
  else if(mode==='rev'||mode==='revDone'){{drawRev(tick);}}
  else{{drawIdle();}}
}}

function drawIdle(){{
  imgBox(sWav,img_x,sY,IW,IH,1,'secret_wav','[B,12,H/2,W/2]','#3b82f6');
  imgBox(cWav,img_x,cY,IW,IH,1,'cover_wav','[B,12,H/2,W/2]','#ec4899');
  arrow(img_x+IW+2,sY+IH/2,blk_x,midY,'#3b82f6',.3,true);
  arrow(img_x+IW+2,cY+IH/2,blk_x,midY,'#ec4899',.3,true);
  for(let i=0;i<NB;i++){{
    blk(blk_x+i*(BW+BGAP),midY-BH/2,BW,BH,''+(i+1),'#2a2d3a',.9,0);
    if(i<NB-1)arrow(blk_x+i*(BW+BGAP)+BW,midY,blk_x+(i+1)*(BW+BGAP),midY,'#2a2d3a',.3);
  }}
  arrow(blk_x+BLK_TOTAL+8,midY,iwt_x,midY,'#2a2d3a',.3);
  imgBox(stegoOff,iwt_x,midY-IH/2,IW,IH,.5,'Stego image','[B,3,H,W]','#10b981');
  txt('Replay buttons below to re-animate',VW/2,VH-10,10,'#374151');
}}

const FP=['Images entering pipeline','DWT: decomposing into wavelet sub-bands (LL,HL,LH,HH)','Concatenating into 24-channel tensor [B,24,H/2,W/2]','Concealing blocks 1-6: Phi-module shifts cover features','Concealing blocks 7-11: rho-module exp(sigma) scaling','Concealing blocks 12-16: eta-module mixes channels','IWT: wavelet domain back to pixel space','Stego complete — secret hidden in cover'];

function drawFwd(f){{
  const ph=f<30?0:f<70?1:f<95?2:f<140?3:f<180?4:f<210?5:f<225?6:7;
  setPD(ph);setPL(FP[ph]);
  const p0=ease(pr(f,0,28)),p1=ease(pr(f,28,68)),p2=ease(pr(f,65,92));
  const p3=ease(pr(f,90,175)),p4=ease(pr(f,170,200)),p5=ease(pr(f,200,228)),p6=ease(pr(f,225,240));

  const dy=(1-p0)*-28;
  imgBox(sOff,img_x,sY+dy,IW,IH,p0,'Secret image','(B,3,H,W)','#3b82f6');
  imgBox(cOff,img_x,cY+dy,IW,IH,p0,'Cover image','(B,3,H,W)','#ec4899');
  arrow(img_x+IW,sY+IH/2,dwt_x,sY+IH/2,'#3b82f6',p0);
  arrow(img_x+IW,cY+IH/2,dwt_x,cY+IH/2,'#ec4899',p0);

  if(p1>0){{
    X.save();X.globalAlpha=p1;X.beginPath();X.rect(dwt_x,sY,IW*p1,IH);X.clip();
    imgBox(sWav,dwt_x,sY,IW,IH,1,'DWT output','[B,12,H/2,W/2]','#3b82f6');X.restore();
    if(p1<1){{X.save();X.globalAlpha=p1*.85;X.strokeStyle='#3b82f6';X.lineWidth=2;X.beginPath();X.moveTo(dwt_x+IW*p1,sY);X.lineTo(dwt_x+IW*p1,sY+IH);X.stroke();X.restore();}}
    X.save();X.globalAlpha=p1;X.beginPath();X.rect(dwt_x,cY,IW*p1,IH);X.clip();
    imgBox(cWav,dwt_x,cY,IW,IH,1,'DWT output','[B,12,H/2,W/2]','#ec4899');X.restore();
    if(p1<1){{X.save();X.globalAlpha=p1*.85;X.strokeStyle='#ec4899';X.lineWidth=2;X.beginPath();X.moveTo(dwt_x+IW*p1,cY);X.lineTo(dwt_x+IW*p1,cY+IH);X.stroke();X.restore();}}
    arrow(dwt_x+IW,sY+IH/2,cat_bx-6,sY+IH/2,'#3b82f6',p1);
    arrow(dwt_x+IW,cY+IH/2,cat_bx-6,cY+IH/2,'#ec4899',p1);
  }}

  if(p2>0){{
    X.save();X.globalAlpha=p2*.65;X.strokeStyle='#374151';X.lineWidth=1.2;X.setLineDash([4,4]);
    X.beginPath();
    X.moveTo(cat_bx,sY+IH/2);X.lineTo(cat_bx+14,sY+IH/2);
    X.lineTo(cat_bx+14,cY+IH/2);X.lineTo(cat_bx,cY+IH/2);
    X.stroke();X.setLineDash([]);X.restore();
    txt('cat',cat_bx+26,midY-8,10,'#6b7280');
    txt('[B,24,H/2,W/2]',cat_bx+26,midY+8,9,'#4b5563');
    arrow(cat_bx+52,midY,blk_x-2,midY,'#6b7280',p2);
  }}

  if(p3>0){{
    const active=Math.min(Math.floor(p3*NB),NB-1);
    for(let i=0;i<NB;i++){{
      const bx=blk_x+i*(BW+BGAP),by=midY-BH/2;
      const done=i<active,isA=i===active;
      const col=done?'#10b981':isA?'#8b5cf6':'#2a2d3a';
      blk(bx,by,BW,BH,''+(i+1),col,1,isA?(tick%24)/24:0);
      if(isA&&sOff&&cOff){{
        X.save();X.beginPath();X.roundRect(bx+1,by+1,BW-2,BH-2,4);X.clip();
        X.imageSmoothingEnabled=true;X.imageSmoothingQuality='high';
        X.globalAlpha=.42;X.drawImage(sOff,bx+1,by+1,BW-2,BH-2);
        X.globalAlpha=.32;X.drawImage(cOff,bx+1,by+1,BW-2,BH-2);X.restore();
      }}
      if(i<NB-1)arrow(bx+BW,midY,bx+BW+BGAP,midY,done?'#10b981':'#2a2d3a',done?.85:.3);
    }}
    const mi=Math.floor((tick%72)/24);
    const mods=['Phi-module: shifts cover features','rho-module: exp(sigma) scaling','eta-module: mixes channels'];
    const mc=['#60a5fa','#f59e0b','#f472b6'];
    txt(mods[mi],blk_x+BLK_TOTAL/2,midY+BH/2+22,11,mc[mi]);
  }}

  if(p4>0){{
    X.save();X.globalAlpha=p4*.75;X.strokeStyle='#374151';X.lineWidth=1.2;X.setLineDash([4,4]);
    X.beginPath();
    X.moveTo(split_x,midY);X.lineTo(split_x+12,stegoForkY);
    X.moveTo(split_x,midY);X.lineTo(split_x+12,zForkY);
    X.stroke();X.setLineDash([]);X.restore();
    txt('stego_wav',split_x+62,stegoForkY-16,10,'#10b981');
    txt('[B,12,H/2,W/2]',split_x+62,stegoForkY-4,8,'#4b5563');
    txt('z  (lost info r)',split_x+62,zForkY+14,10,'#f59e0b');
    txt('[B,12,H/2,W/2]',split_x+62,zForkY+26,8,'#4b5563');
    arrow(split_x+12,stegoForkY,iwt_x,stegoForkY,'#10b981',p4);
  }}

  if(p5>0){{
    X.save();X.globalAlpha=p5;X.beginPath();X.rect(iwt_x,iwtY,IW*p5,IH);X.clip();
    imgBox(stegoOff,iwt_x,iwtY,IW,IH,1,'IWT','pixel domain','#10b981');X.restore();
    if(p5<1){{X.save();X.globalAlpha=p5*.9;X.strokeStyle='#10b981';X.lineWidth=2;X.beginPath();X.moveTo(iwt_x+IW*p5,iwtY);X.lineTo(iwt_x+IW*p5,iwtY+IH);X.stroke();X.restore();}}
    arrow(iwt_x+IW,iwtY+IH/2,out_x,iwtY+IH/2,'#10b981',p5);
    imgBox(stegoOff,out_x,iwtY,IW,IH,p5,'Stego image','(B,3,H,W)','#10b981',p5>.7);
    if(p5>.3&&sOff){{
      X.save();X.globalAlpha=p5*.1;X.beginPath();X.roundRect(out_x,iwtY,IW,IH,6);X.clip();
      X.drawImage(sOff,out_x,iwtY,IW,IH);X.restore();
    }}
  }}
  if(p6>0){{
    txt('Secret hidden inside cover  |  PSNR >30 dB',out_x+IW/2,iwtY+IH+20,10,'#10b981');
  }}
}}

const RP=['Stego + Gaussian z entering reveal pass','DWT applied to stego image','Concatenated 24-channel input to reversed blocks','Reversed blocks 16-12: eta inverse','Reversed blocks 11-7: rho inverse','Reversed blocks 6-1: phi inverse — secret_wav recovered','IWT reconstructing secret from wavelet sub-bands','Secret recovered successfully'];

function drawRev(f){{
  const ph=f<30?0:f<65?1:f<90?2:f<140?3:f<180?4:f<215?5:f<232?6:7;
  setPD(ph);setPL(RP[ph]);
  const p0=ease(pr(f,0,28)),p1=ease(pr(f,28,65)),p2=ease(pr(f,62,90));
  const p3=ease(pr(f,88,175)),p4=ease(pr(f,172,202)),p5=ease(pr(f,200,230)),p6=ease(pr(f,228,240));

  imgBox(stegoOff,out_x,sY,IW,IH,p0,'Stego input','(B,3,H,W)','#10b981',p0>.5);
  arrow(out_x,sY+IH/2,iwt_x+IW,sY+IH/2,'#10b981',p0);

  if(p1>0){{
    X.save();X.globalAlpha=p1;X.beginPath();X.rect(iwt_x,sY,IW,IH);X.clip();
    imgBox(cWav,iwt_x,sY,IW,IH,1,'DWT stego','[B,12,H/2,W/2]','#10b981');X.restore();
    arrow(iwt_x,sY+IH/2,blk_x+BLK_TOTAL+2,midY,'#10b981',p1);
  }}

  if(p2>0){{
    const bx=blk_x+BLK_TOTAL+6;
    X.save();X.globalAlpha=p2*.65;X.strokeStyle='#374151';X.lineWidth=1.2;X.setLineDash([4,4]);
    X.beginPath();X.moveTo(bx,midY-16);X.lineTo(bx+12,midY-16);X.lineTo(bx+12,midY+14);X.lineTo(bx,midY+14);
    X.stroke();X.setLineDash([]);X.restore();
    txt('cat [24ch]',bx+38,midY,10,'#6b7280');
    arrow(bx+12,midY,bx-2,midY,'#6b7280',p2);
  }}

  if(p3>0){{
    const active=NB-1-Math.min(Math.floor(p3*NB),NB-1);
    for(let i=NB-1;i>=0;i--){{
      const bx=blk_x+i*(BW+BGAP),by=midY-BH/2;
      const done=i>active,isA=i===active;
      const col=done?'#3b82f6':isA?'#8b5cf6':'#2a2d3a';
      blk(bx,by,BW,BH,''+(i+1),col,1,isA?(tick%24)/24:0);
      if(isA&&cOff&&sOff){{
        X.save();X.beginPath();X.roundRect(bx+1,by+1,BW-2,BH-2,4);X.clip();
        X.imageSmoothingEnabled=true;X.imageSmoothingQuality='high';
        X.globalAlpha=.35;X.drawImage(cOff,bx+1,by+1,BW-2,BH-2);
        X.globalAlpha=.35;X.drawImage(sOff,bx+1,by+1,BW-2,BH-2);X.restore();
      }}
      if(i>0)arrow(bx,midY,bx-BGAP,midY,done?'#3b82f6':'#2a2d3a',done?.85:.3);
    }}
    const mi=Math.floor((tick%72)/24);
    const mods=['eta inverse: undo eta','rho inverse: undo scaling','phi inverse: undo shift'];
    const mc=['#f472b6','#f59e0b','#60a5fa'];
    txt(mods[mi],blk_x+BLK_TOTAL/2,midY+BH/2+22,11,mc[mi]);
  }}

  if(p4>0){{
    const sx=blk_x-12;
    X.save();X.globalAlpha=p4*.75;X.strokeStyle='#374151';X.lineWidth=1.2;X.setLineDash([4,4]);
    X.beginPath();X.moveTo(sx+12,midY);X.lineTo(sx,sY+IH/2);X.moveTo(sx+12,midY);X.lineTo(sx,cY+IH/2);
    X.stroke();X.setLineDash([]);X.restore();
    arrow(sx,sY+IH/2,dwt_x+IW,sY+IH/2,'#3b82f6',p4);
    arrow(sx,cY+IH/2,dwt_x+IW,cY+IH/2,'#6b7280',p4*.5);
    txt('secret_wav',dwt_x+IW+44,sY+IH/2-12,10,'#3b82f6');
    txt('cover_wav', dwt_x+IW+42,cY+IH/2+12,10,'#6b7280');
  }}

  if(p5>0){{
    X.save();X.globalAlpha=p5;X.beginPath();X.rect(dwt_x,sY,IW*p5,IH);X.clip();
    imgBox(recOff,dwt_x,sY,IW,IH,1,'IWT','pixel domain','#3b82f6');X.restore();
    if(p5<1){{X.save();X.globalAlpha=p5*.9;X.strokeStyle='#3b82f6';X.lineWidth=2;X.beginPath();X.moveTo(dwt_x+IW*p5,sY);X.lineTo(dwt_x+IW*p5,sY+IH);X.stroke();X.restore();}}
    arrow(dwt_x,sY+IH/2,img_x+IW,sY+IH/2,'#3b82f6',p5);
    imgBox(recOff,img_x,sY,IW,IH,p5,'Recovered secret','(B,3,H,W)','#3b82f6',p5>.7);
  }}

  if(p6>0){{
    imgBox(sOff, img_x,        cY,IW,IH,p6,'Original','secret','#6b7280');
    imgBox(recOff,img_x+IW+10,cY,IW,IH,p6,'Recovered','from stego','#3b82f6',true);
    txt('vs',img_x+IW+5,cY+IH/2,12,'#374151');
  }}
}}

const TOTAL=240;
function loop(){{
  tick+=spd;if(tick>TOTAL)tick=TOTAL;draw();
  if(tick<TOTAL){{raf=requestAnimationFrame(loop);}}
  else{{
    animating=false;
    if(mode==='fwd'){{mode='fwdDone';setPL('Forward pass complete');}}
    if(mode==='rev'){{mode='revDone';setPL('Reveal pass complete — secret recovered!');}}
  }}
}}
function startFwd(){{
  if(animating)return;animating=true;tick=0;mode='fwd';
  if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(loop);
}}
function startRev(){{
  if(animating)return;animating=true;tick=0;mode='rev';
  document.getElementById('bR').disabled=true;
  if(raf)cancelAnimationFrame(raf);raf=requestAnimationFrame(loop);
}}
function reset(){{
  if(raf)cancelAnimationFrame(raf);animating=false;tick=0;mode='idle';
  document.getElementById('bR').disabled=false;
  for(let i=0;i<8;i++)document.getElementById('d'+i).className='pd';
  setPL('Idle');draw();
}}
document.getElementById('bR').onclick=startRev;
document.getElementById('bX').onclick=reset;
window.addEventListener('resize',()=>{{if(raf)cancelAnimationFrame(raf);raf=null;draw();if(animating)raf=requestAnimationFrame(loop);}});
</script></body></html>"""


# ─────────────────────────────────────────────
# Session state — pipeline_mode is ONLY "hidden" or "idle"
# It NEVER changes to "rev" — the reveal section always shows
# once stego is generated, regardless of which attack is selected.
# ─────────────────────────────────────────────
for k in ["stego_pil","stego_t","out_steg","out_z","pipeline_mode",
          "secret_b64","cover_b64","stego_b64",
          "recovered_pil","attacked_pil","recovered_b64",
          "rec_psnr","atk_psnr","atk_note","last_attack","just_revealed","stego_cover_psnr"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Model Status")
    net, load_msg = load_model()
    if net is not None:
        st.success("hinet_final.pth loaded")
        with st.expander("Load details"): st.code(load_msg)
    else:
        st.error("Model failed to load"); st.code(load_msg)
        st.info("Place `hinet_final.pth` in a `checkpoints/` folder next to `app.py`.")
    st.markdown("---")

# ─────────────────────────────────────────────
# Main UI — three clear sections that never collapse
# ─────────────────────────────────────────────
st.markdown("## Upload Images")
c1, c2 = st.columns(2)
with c1:
    cover_file = st.file_uploader("Cover Image", type=["png","jpg","jpeg"])
    if cover_file:
        st.image(cover_file, caption="Cover", width=310)
with c2:
    secret_file = st.file_uploader("Secret Image", type=["png","jpg","jpeg"])
    if secret_file:
        st.image(secret_file, caption="Secret", width=310)

if cover_file and secret_file:
    cover  = Image.open(cover_file).convert("RGB")
    secret = Image.open(secret_file).convert("RGB")

    cover_b64  = pil_to_b64(cover.resize((256,256)))
    secret_b64 = pil_to_b64(secret.resize((256,256)))
    st.session_state.cover_b64  = cover_b64
    st.session_state.secret_b64 = secret_b64

    # ══════════════════════════════════════════
    # SECTION 1 — Hide
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## Hide Secret Image in Cover")
    st.caption("Runs the model forward pass and animates the concealing architecture.")

    if st.button("Hide Secret Image", disabled=(net is None)):
        with st.spinner("Running HiNetModel forward pass…"):
            try:
                stego_pil, out_steg, out_z = run_hide(net, cover, secret)
                st.session_state.stego_pil     = stego_pil
                st.session_state.stego_t       = pil2t(stego_pil)
                st.session_state.out_steg      = out_steg
                st.session_state.out_z         = out_z
                st.session_state.stego_b64     = pil_to_b64(stego_pil.resize((256,256)))
                st.session_state.pipeline_mode = "hidden"
                # Reset any previous reveal results so old images don't linger
                st.session_state.recovered_pil  = None
                st.session_state.attacked_pil   = None
                st.session_state.recovered_b64  = None
                st.session_state.rec_psnr       = None
                st.session_state.atk_psnr       = None
                st.session_state.last_attack    = None
                st.success("Stego image created — forward pass animating below!")
            except Exception as e:
                st.error(f"Hiding failed: {e}"); st.exception(e)

    # Forward animation + stego image — visible whenever stego exists
    if st.session_state.pipeline_mode == "hidden" and st.session_state.stego_b64:
        cover_arr = np.array(cover.resize((IMG_SIZE,IMG_SIZE)))
        stego_arr = np.array(st.session_state.stego_pil.resize((IMG_SIZE,IMG_SIZE)))
        psnr_val  = psnr(cover_arr, stego_arr)
        st.session_state.stego_cover_psnr = psnr_val

        components.html(
            build_pipeline_html(
                secret_b64=st.session_state.secret_b64,
                cover_b64 =st.session_state.cover_b64,
                auto_mode ="fwd",
                stego_b64 =st.session_state.stego_b64,
            ),
            height=360, scrolling=False
        )

        st.markdown("**Generated Stego Image**")
        g1, g2, _ = st.columns([1,1,2])
        with g1:
            st.image(st.session_state.stego_pil, width=280,
                     caption=f"Generated stego — PSNR vs cover: {psnr_val:.1f} dB")

        # ══════════════════════════════════════════
        # SECTION 2 — Attack + Reveal
        # ══════════════════════════════════════════
        st.markdown("---")
        st.markdown("## Apply Attack & Recover Secret")
        st.caption("Select an attack and click the button. Switching attacks clears previous results.")

        # FIX: on_change clears all previous results so old output never persists
        def _clear_results():
            st.session_state.recovered_pil  = None
            st.session_state.attacked_pil   = None
            st.session_state.recovered_b64  = None
            st.session_state.rec_psnr       = None
            st.session_state.atk_psnr       = None
            st.session_state.atk_note       = None
            st.session_state.last_attack    = None

        selected = st.selectbox(
            "Attack type",
            list(ATTACKS.keys()),
            index=0,
            key="attack_select",
            on_change=_clear_results   # clears results immediately on dropdown change
        )
        atk_fn, is_clean, desc = ATTACKS[selected]
        st.caption(f"_{desc}_")

        if st.button("Apply Attack & Reveal Secret", disabled=(net is None)):
            with st.spinner(f"Applying '{selected}' and recovering secret…"):
                try:
                    attacked_t   = atk_fn(st.session_state.stego_t)
                    attacked_pil = t2pil(attacked_t)

                    stego_arr2   = np.array(st.session_state.stego_pil.resize((IMG_SIZE,IMG_SIZE)))
                    attacked_arr = np.array(attacked_pil.resize((IMG_SIZE,IMG_SIZE)))
                    atk_psnr_val = st.session_state.stego_cover_psnr if is_clean else psnr(stego_arr2, attacked_arr)
                    atk_note = " — PSNR vs cover (clean/no-attack)" if is_clean else ""

                    recovered = run_reveal(
                        net, attacked_t,
                        st.session_state.out_steg,
                        st.session_state.out_z,
                        is_clean=is_clean
                    )
                    secret_rs    = secret.resize((IMG_SIZE,IMG_SIZE))
                    rec_arr      = np.array(recovered.resize((IMG_SIZE,IMG_SIZE)))
                    rec_psnr_val = psnr(np.array(secret_rs), rec_arr)

                    st.session_state.recovered_pil  = recovered
                    st.session_state.attacked_pil   = attacked_pil
                    st.session_state.recovered_b64  = pil_to_b64(recovered.resize((256,256)))
                    st.session_state.rec_psnr        = rec_psnr_val
                    st.session_state.atk_psnr        = atk_psnr_val
                    st.session_state.atk_note        = atk_note
                    st.session_state.last_attack     = selected
                    st.session_state.just_revealed   = True   # flag: animate this render only
                    st.success(f"Recovery complete for: {selected}")
                except Exception as e:
                    st.error(f"Failed: {e}"); st.exception(e)

        # Results shown ONLY when last_attack matches the currently selected attack.
        # Switching the dropdown triggers _clear_results → last_attack becomes None
        # → this block is hidden until the button is clicked for the new attack.
        if st.session_state.recovered_b64 and st.session_state.last_attack == selected:
            st.markdown(f"**Results for: {st.session_state.last_attack}**")

            # auto_mode="rev" only on the render immediately after button click.
            # On all subsequent renders (e.g. scrolling, dropdown hover) it is "idle"
            # so the animation doesn't restart by itself.
            rev_auto = "rev" if st.session_state.get("just_revealed") else "idle"
            st.session_state.just_revealed = False  # consume the flag

            # Reveal pass animation
            components.html(
                build_pipeline_html(
                    secret_b64    =st.session_state.secret_b64,
                    cover_b64     =st.session_state.cover_b64,
                    auto_mode     =rev_auto,
                    stego_b64     =st.session_state.stego_b64,
                    recovered_b64 =st.session_state.recovered_b64,
                ),
                height=380, scrolling=False
            )

            # 4-column image comparison
            st.markdown("#### Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.image(st.session_state.stego_pil,
                     caption="Stego (clean)", use_container_width=True)
            r2.image(st.session_state.attacked_pil,
                     caption=f"After: {st.session_state.last_attack}", use_container_width=True)
            r3.image(secret.resize((IMG_SIZE,IMG_SIZE)),
                     caption="Original Secret", use_container_width=True)
            r4.image(st.session_state.recovered_pil,
                     caption="Recovered Secret", use_container_width=True)

            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Attack PSNR  (stego vs attacked)",
                    f"{st.session_state.atk_psnr:.1f} dB",
                    help="Higher = lighter attack. 60 dB = identical images."
                )
                if st.session_state.atk_note:
                    st.caption(f"{st.session_state.atk_psnr:.1f} dB {st.session_state.atk_note}")
            with m2:
                st.metric(
                    "Recovery PSNR  (secret vs recovered)",
                    f"{st.session_state.rec_psnr:.1f} dB"
                )
                rpv = st.session_state.rec_psnr
                if   rpv > 30: st.success("Excellent recovery!")
                elif rpv > 20: st.warning("Good — try a lighter attack.")
                else:          st.error("Poor recovery — attack too destructive.")

else:
    st.info("Upload both a cover image and a secret image to get started.")
