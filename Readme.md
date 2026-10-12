# คู่มือ Setup เครื่อง Jetson ใหม่ (JetPack 5.1.3 / L4T 35.5.0)
https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

คู่มือนี้ใช้ติดตั้ง PyTorch (ใช้ GPU ได้), torchvision, ultralytics และ dependencies ของโปรเจกต์ ตั้งแต่เครื่องใหม่จนพร้อมรัน

**ค่าของระบบนี้ (ต้องตรงกัน ไม่งั้น wheel ลงไม่ได้)**

| รายการ | ค่า |
|---|---|
| JetPack | 5.1.3 |
| L4T | 35.5.0 |
| Ubuntu | 20.04 |
| Python | 3.8 (`/usr/bin/python3`) |
| CUDA | 11.4 |
| สถาปัตยกรรม | aarch64 |
| torch | 2.1.0a0+41361538.nv23.06 (wheel ของ NVIDIA) |
| torchvision | 0.16.1 (build เอง) |
| numpy | 1.23.5 |

> ห้ามลง torch จาก PyPI หรือ `download.pytorch.org` บน Jetson เพราะจะได้ตัวที่ใช้ GPU ไม่ได้

---

## 0. ตรวจสอบเครื่องก่อนเริ่ม

```bash
cat /etc/nv_tegra_release        # ต้องเห็น R35 (release), REVISION: 5.0
uname -m                         # ต้องได้ aarch64
/usr/bin/python3 --version       # ต้องได้ Python 3.8.x
```

---

## 1. อัปเดตระบบและซ่อม apt

```bash
sudo apt update
sudo dpkg --configure -a
sudo apt --fix-broken install
sudo apt upgrade -y
```

ถ้าขึ้น `N not fully installed or removed` ให้รัน `dpkg --configure -a` และ `--fix-broken install` ซ้ำจนไม่มี error ก่อนไปขั้นต่อไป

---

## 2. ติดตั้ง JetPack components (CUDA, cuDNN, TensorRT)

```bash
sudo apt install -y nvidia-jetpack
```

ตั้งค่า PATH ของ CUDA ให้ถาวร:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version                   # ต้องเห็น release 11.4
```

---

## 3. (แนะนำ) ตั้ง power mode และเพิ่ม swap

```bash
sudo nvpmodel -m 0               # โหมดแรงสุด (บางรุ่นเลขโหมดต่างกัน ดู sudo nvpmodel -q)
sudo jetson_clocks               # ล็อกความเร็ว clock
```

ขั้นตอน build torchvision กิน RAM มาก ถ้าเครื่อง RAM 4-8 GB แนะนำเพิ่ม swap:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h
```

---

## 4. ติดตั้ง system packages

```bash
sudo apt install -y \
  git cmake build-essential \
  python3-pip python3-venv python3-dev libpython3-dev \
  libopenblas-base libopenblas-dev libopenmpi-dev libomp-dev \
  libjpeg-dev zlib1g-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libturbojpeg0-dev
```

(`libturbojpeg0-dev` จำเป็นสำหรับ PyTurboJPEG)

ติดตั้ง jtop สำหรับดู GPU/RAM/อุณหภูมิ (ลงด้วย pip ของระบบ ไม่ใช่ใน venv):

```bash
sudo pip3 install -U jetson-stats
sudo reboot                      # ต้องรีบูตหลังลง jtop ครั้งแรก
```

---

## 5. สร้าง virtual environment

ใช้ `--system-site-packages` เพื่อให้ venv มองเห็น TensorRT และ OpenCV ที่มากับ JetPack

```bash
/usr/bin/python3 -m venv --system-site-packages ~/.venv
source ~/.venv/bin/activate
pip install --upgrade pip wheel setuptools
```

ทุกครั้งที่เปิด terminal ใหม่ต้อง `source ~/.venv/bin/activate` (หรือใส่ไว้ท้าย `~/.bashrc`)

---

## 6. ติดตั้ง numpy และ PyTorch

```bash
pip install numpy==1.23.5

pip install --no-cache https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

ทดสอบ:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

ต้องได้ `2.1.0a0+41361538.nv23.06 True`

ถ้า Illegal instruction ให้เพิ่มบรรทัดนี้ใน `~/.bashrc`:

```bash
export OPENBLAS_CORETYPE=ARMV8
```

---

## 7. Build torchvision (ใช้เวลาประมาณ 10-30 นาที)

```bash
cd ~
git clone --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.16.1
python3 setup.py bdist_wheel
pip install dist/torchvision-0.16.1*.whl
cd ~
```

ทดสอบ (ต้อง `cd ~` ออกจากโฟลเดอร์ torchvision ก่อน ไม่งั้น import ผิดที่):

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

**เก็บ wheel ไว้ใช้กับเครื่องใหม่ในอนาคต** (ต้องเป็น JetPack และ Python เวอร์ชันเดียวกัน):

```bash
mkdir -p ~/wheels
cp ~/torchvision/dist/torchvision-0.16.1*.whl ~/wheels/
```

---

## 8. ติดตั้ง dependencies ของโปรเจกต์

ใช้ไฟล์ `requirements.txt` ที่แก้แล้ว (ไม่มี torch/torchvision/torchaudio และไม่มี cu121 index)

```bash
cd <โฟลเดอร์โปรเจกต์>
pip install -r requirements.txt
```

**เช็กทันทีว่า torch ไม่ถูกทับ:**

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

ต้องยังเป็น `2.1.0a0+41361538.nv23.06 0.16.1 True` ถ้าเปลี่ยนไป ให้ลงใหม่:

```bash
pip uninstall -y torch torchvision
# แล้วทำขั้นตอนที่ 6 และ 7 ใหม่
```

---

## 9. ทดสอบทั้งระบบ

```bash
python - <<'EOF'
import torch, torchvision, numpy, cv2
print("torch      :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("device     :", torch.cuda.get_device_name(0))
print("torchvision:", torchvision.__version__)
print("numpy      :", numpy.__version__)
print("opencv     :", cv2.__version__)

x = torch.randn(1000, 1000, device="cuda")
print("gpu matmul :", (x @ x).sum().item() != 0)

from ultralytics import YOLO
print("ultralytics: OK")

from turbojpeg import TurboJPEG
TurboJPEG()
print("turbojpeg  : OK")
EOF
```

---

## 10. ปัญหาที่พบบ่อย

| อาการ | สาเหตุ / วิธีแก้ |
|---|---|
| `not a supported wheel on this platform` | Python ไม่ใช่ 3.8 หรือไม่ใช่ aarch64 เช็ก `python --version` และ `uname -m` สร้าง venv ใหม่ด้วย `/usr/bin/python3` |
| `torch.cuda.is_available()` เป็น False | ลง torch จาก PyPI ผิดตัว ให้ `pip uninstall torch` แล้วลง wheel ของ NVIDIA ใหม่ |
| torch เปลี่ยนเวอร์ชันหลัง `pip install -r` | มีแพ็กเกจไปดึง torch ตัวอื่นมาทับ ลง wheel ของ NVIDIA ซ้ำ (`--force-reinstall --no-deps`) |
| `libcudnn.so.8` / `libcusparseLt` หาไม่เจอ | ยังไม่ได้ลง `nvidia-jetpack` หรือยังไม่ตั้ง `LD_LIBRARY_PATH` |
| `Illegal instruction (core dumped)` | `export OPENBLAS_CORETYPE=ARMV8` |
| import พังเพราะ numpy | ต้องเป็น numpy 1.x (`pip install numpy==1.23.5`) |
| `torchvision::nms does not exist` | torchvision ไม่ตรงกับ torch ต้อง build v0.16.1 ใหม่ |
| `import torchvision` ผิดปกติตอนอยู่ในโฟลเดอร์ torchvision | ให้ `cd ~` แล้วรันใหม่ |
| PyTurboJPEG หา lib ไม่เจอ | `sudo apt install libturbojpeg0-dev` |
| `onnx-graphsurgeon requires onnx` | เป็นแค่คำเตือน ข้ามได้ ถ้าจะใช้จริงค่อย `pip install onnx` |
| build torchvision แล้วเครื่องค้าง/ถูก kill | RAM ไม่พอ เพิ่ม swap ตามขั้นตอนที่ 3 |
| กล้อง GStreamer เปิดไม่ได้ใน OpenCV | `opencv-python` จาก pip ไม่มี GStreamer ให้ใช้ OpenCV ของระบบ: `pip uninstall opencv-python opencv-python-headless` แล้วใช้ `cv2` จาก JetPack ผ่าน `--system-site-packages` |

---

## 11. (ไม่บังคับ) Export โมเดลเป็น TensorRT เพื่อให้เร็วขึ้น

```bash
yolo export model=yolov8n.pt format=engine half=True device=0
```

ต้องใช้ venv แบบ `--system-site-packages` เพราะ TensorRT python binding อยู่ในระบบ

---

## เช็กลิสต์สรุป

- [ ] `nvcc --version` ได้ 11.4
- [ ] `/usr/bin/python3 --version` ได้ 3.8.x
- [ ] เปิด venv แล้ว (`(.venv)` นำหน้า prompt)
- [ ] `torch.cuda.is_available()` เป็น True
- [ ] `torchvision.__version__` เป็น 0.16.1
- [ ] `pip install -r requirements.txt` ผ่านและ torch ไม่ถูกทับ
- [ ] สคริปต์ทดสอบในขั้นตอนที่ 9 ผ่านทุกบรรทัด