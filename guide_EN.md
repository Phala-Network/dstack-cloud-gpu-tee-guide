# Deploy dstack-cloud GPU TEE (H100 Confidential Compute) on GCP

A from-scratch walkthrough for deploying a Confidential Compute GPU
workload on Google Cloud: Intel TDX guest + NVIDIA H100 in **Hopper
Confidential Compute (HCC) mode**, with an attested key handshake
through dstack-kms and a sample PyTorch container.

By the end you should have:

- A spot `a3-highgpu-1g` instance booted from a custom dstack TEE image.
- An H100 GPU initialized with `CC status: ON` and a live SPDM session.
- SSH access to the guest through the dstack gateway.
- A PyTorch CUDA matmul running inside Docker, on the H100.

If you only need a non-GPU dstack-cloud deployment (e.g. running KMS
itself, or a CPU-only TEE workload), follow
[`Phala-Network/dstack-cloud-deployment-guide`](https://github.com/Phala-Network/dstack-cloud-deployment-guide)
instead — this guide focuses on the GPU-specific bits.

---

## 1. Architecture and Goals

```
┌──────────────────────────────────────────────────────────────┐
│  Your laptop / dev box                                       │
│  - gcloud SDK + dstack-cloud CLI                             │
│  - dstack-cloud-nvidia-0.6.1 OS image (boot disk)            │
└──────────────────────────────┬───────────────────────────────┘
                               │ gcloud compute instances create
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  GCP project                                                 │
│  zone: us-central1-{a,b,c} / us-west1 / europe-west1 / ...   │
│  machine: a3-highgpu-1g (SPOT, TDX, TERMINATE)               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ TDX guest (dstack-cloud-nvidia-0.6.1)                  │  │
│  │  - Linux 6.18.7-dstack with CONFIG_CRYPTO_ECDSA=y      │  │
│  │  - nvidia.ko built with USE_LKCA → libspdm over LKCA   │  │
│  │  - dstack-prepare → KMS handshake → LUKS data disk     │  │
│  │  - nvidia-persistenced --uvm-persistence-mode          │  │
│  │  - nvidia-smi conf-compute -srs 1 (set ready state)    │  │
│  │  - dstack guest agent + dstack gateway tunnel          │  │
│  │  - app-compose: your Docker workload                   │  │
│  └─────────────────────┬──────────────────────────────────┘  │
└────────────────────────┼─────────────────────────────────────┘
                         │ SPDM (ECDSA/ECDH/HMAC over LKCA)
                         ▼
                ┌────────────────┐
                │ H100 80GB HBM3 │  CC mode = ON
                └────────────────┘
```

**Why we ship a custom image.** The default GCP confidential-VM guide
calls for installing the NVIDIA driver against
[`nvidia-driver-575-open`](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/create-a-confidential-vm-instance-with-gpu)
plus an LKCA modprobe shim. dstack's TEE image is read-only and
verity-protected, so we bake the driver, Linux Kernel Crypto API
support, and the persistence/ready-state services into the image
itself. The kernel must be built with `CONFIG_CRYPTO_ECDSA=y` for the
NVIDIA driver to define `USE_LKCA`; otherwise `libspdm` falls back to
stubs and SPDM never completes (see
[meta-dstack-cloud#14](https://github.com/Phala-Network/meta-dstack-cloud/pull/14)).
This is fixed starting with **`dstack-cloud-nvidia-0.6.1`**.

---

## 2. Prerequisites

### 2.1 Tools

You'll need these on your local machine:

| Tool                                            | Tested version | Notes |
| ----------------------------------------------- | -------------- | ----- |
| [gcloud SDK](https://cloud.google.com/sdk/docs/install) | 565+        | `gcloud auth login` against your GCP account |
| [`dstack-cloud`](https://github.com/Phala-Network/meta-dstack-cloud/blob/main/scripts/bin/dstack-cloud) | latest from `main` | single Python file, drop on `$PATH` |
| `openssl`                                       | any            | SSH-over-TLS proxy command |
| `curl`, `jq`, `tar`                             | any            | |

### 2.2 GCP Project, billing, H100 quota

You need a project with:

1. **Billing** enabled.
2. **Compute Engine API** enabled (`gcloud services enable compute.googleapis.com`).
3. **GCS bucket** for staging the boot/shared disk (e.g. `gs://<project>-dstack`).
4. **H100 quota.** On a fresh project, on-demand H100 (`NVIDIA_H100_GPUS`) is
   typically `0`. Confirm what you do have:

   ```bash
   gcloud beta quotas info list \
     --service=compute.googleapis.com \
     --project=<project> \
     --filter='quotaId:H100' --format=json \
     | jq -r '.[].quotaId'
   ```

   On most projects you'll only see `PREEMPTIBLE-NVIDIA-H100-GPUS-...`
   (i.e. **spot** is the only path). This guide deploys to spot. If
   you have on-demand quota, drop `--provisioning-model=SPOT` later.

H100-capable zones (as of 2026-05): `us-central1-a/b/c`, `us-west1-a/b`,
`europe-west1-b/c`, `asia-east1-c`, `asia-northeast1-b`,
`asia-southeast1-b`. Verify your target zone exposes the accelerator
type:

```bash
gcloud compute accelerator-types list --filter='name~h100' \
  --format='table(name,zone)' | grep <zone>
```

### 2.3 Install `dstack-cloud`

```bash
mkdir -p ~/.local/bin
curl -fsSL -o ~/.local/bin/dstack-cloud \
  https://raw.githubusercontent.com/Phala-Network/meta-dstack-cloud/main/scripts/bin/dstack-cloud
chmod +x ~/.local/bin/dstack-cloud
dstack-cloud --help | head -5
```

> **H100 spot patch (temporary).** The shipped `dstack-cloud` does not
> yet pass `--provisioning-model=SPOT` to `gcloud`. Until that's
> upstream, patch your local copy:
>
> ```bash
> sed -i 's|"--maintenance-policy=TERMINATE",|"--maintenance-policy=TERMINATE",\n            "--provisioning-model=SPOT",\n            "--instance-termination-action=STOP",|' \
>   ~/.local/bin/dstack-cloud
> ```

### 2.4 Configure `dstack-cloud`

Create `~/.config/dstack-cloud/config.json` (replace project/bucket/zone):

```json
{
  "services": {
    "kms_urls":     ["https://kms.tdxlab.dstack.org:13001"],
    "gateway_urls": ["https://gateway.tdxlab.dstack.org:13002"],
    "pccs_url":     ""
  },
  "image_search_paths": [
    "~/.dstack/images"
  ],
  "gcp": {
    "project": "<your-gcp-project>",
    "zone":    "us-central1-a",
    "bucket":  "gs://<your-gcs-bucket>"
  }
}
```

`kms.tdxlab.dstack.org:13001` is a **shared development KMS**. For
production deploy your own KMS first — see the [KMS deployment
guide](https://github.com/Phala-Network/dstack-cloud-deployment-guide).

### 2.5 Download the dstack-cloud-nvidia OS image

```bash
dstack-cloud pull dstack-cloud-nvidia-0.6.1
ls ~/.dstack/images/dstack-cloud-nvidia-0.6.1/
# disk.raw  auth_hash.txt
```

This pulls the UKI variant from
`https://github.com/Phala-Network/meta-dstack-cloud/releases/download/v0.6.1/dstack-cloud-nvidia-0.6.1-uki.tar.gz`
and extracts it under the first entry of `image_search_paths`.

> The image must be **`0.6.1` or later**. `0.6.0` ships a kernel
> without `CONFIG_CRYPTO_ECDSA` and will fail SPDM on H100
> (see [meta-dstack-cloud#14](https://github.com/Phala-Network/meta-dstack-cloud/pull/14)).

### 2.6 KMS allowlist

When you boot a new image hash for the first time, the KMS must accept
its `mr_aggregated` measurement. The shared dev KMS at
`kms.tdxlab.dstack.org:13001` runs with `enforce_self_authorization = false`
and `auth_api.type = "dev"`, so it accepts any well-formed quote — no
on-chain registration required. **For production KMS**, register the
OS image hash on the `DstackKms` contract first; see §3 of the KMS
guide.

---

## 3. Create the Project

### 3.1 `dstack-cloud new`

```bash
dstack-cloud new gpu-hello
cd gpu-hello
ls
# app.json  docker-compose.yaml  prelaunch.sh  .env  .user-config  shared/
```

> **No dots in the instance name.** GCP image names (used for the
> boot disk + shared disk) reject `.`. `dstack-cloud new test-0.6.1`
> will fail at deploy time. Use `gpu-hello`, `h100-demo`, etc.

### 3.2 Edit `app.json`

Replace the `gcp_config` block. Key changes:

- `os_image`: `dstack-cloud-nvidia-0.6.1`
- `machine_type`: `a3-highgpu-1g`
- `data_size`: bump up if you'll load big model weights

```json
{
  "name": "gpu-hello",
  "os_image": "dstack-cloud-nvidia-0.6.1",
  "gcp_config": {
    "project": "",
    "zone": "us-central1-a",
    "instance_name": "dstack-gpu-hello",
    "machine_type": "a3-highgpu-1g",
    "data_size": 100,
    "network": "default"
  },
  "gateway_enabled": true,
  "public_logs": true,
  "key_provider": "kms",
  "storage_fs": "ext4"
}
```

Empty `project`/`bucket` fall back to your global config from §2.4.

### 3.3 `docker-compose.yaml` — PyTorch CUDA smoke test

```yaml
services:
  pytorch:
    image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
    runtime: nvidia
    environment:
      NVIDIA_VISIBLE_DEVICES: all
    command: >
      python -c "
      import torch, time
      print(f'CUDA available: {torch.cuda.is_available()}')
      print(f'Device: {torch.cuda.get_device_name(0)}')
      print(f'Capability: {torch.cuda.get_device_capability(0)}')
      x = torch.randn(4096, 4096, device='cuda')
      y = torch.randn(4096, 4096, device='cuda')
      torch.cuda.synchronize(); t0 = time.time()
      for _ in range(100):
          z = x @ y
      torch.cuda.synchronize()
      dt = time.time() - t0
      print(f'100x 4096^3 matmul: {dt:.3f}s -> {100*2*4096**3/dt/1e9:.0f} GFLOPs')
      "
    restart: "no"
```

This pulls the PyTorch image (~6 GB), runs one CUDA matmul benchmark,
and exits. Replace with your real workload once you have the basics
working.

### 3.4 (Optional) `prelaunch.sh` — SSH installer

Add SSH access to the TEE guest. Edit `prelaunch.sh`:

```bash
#!/bin/sh
set -e
echo "Installing OpenSSH server via dstack-openssh-installer..."
docker run --rm --privileged --pid=host --net=host -v /:/host \
  -e SSH_GITHUB_USER="<your-github-handle>" \
  kvin/dstack-openssh-installer:latest
echo "OpenSSH installation complete"
chmod +x prelaunch.sh
```

`SSH_GITHUB_USER` should be a GitHub handle whose **public keys** you
want to allow. Keys are fetched from `https://github.com/<user>.keys`
at deploy time.

---

## 4. Deploy

### 4.1 Prepare shared files

```bash
dstack-cloud prepare
ls shared/
# .instance_info  .sys-config.json  .user-config  app-compose.json
```

### 4.2 Deploy

```bash
dstack-cloud -v deploy
# ...
# === Deployment Complete ===
# Instance: dstack-gpu-hello
# External IP: <ip>
# Internal IP: <ip>
```

If the deploy fails with `ZONE_RESOURCE_POOL_EXHAUSTED`, retry in a
different H100 zone (edit `app.json` `zone`, run `dstack-cloud remove`
if a partial instance exists, then `dstack-cloud deploy` again).

### 4.3 Watch boot

```bash
dstack-cloud logs -n 20000 | grep -iE \
  'prepare|kms|app keys|nvidia|app-compose|nginx|reboot|fail|panic'
```

You're looking for, in order:

```
dstack-prepare.sh ... Requesting app keys from KMS
dstack-prepare.sh ... Key provider info: KeyProviderInfo { name: "kms", ... }
dstack-prepare.sh ... Setting up disk encryption
...
NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  580.105.08
libspdm_check_crypto_backend: LKCA wrappers found.
...
app-compose.sh ... Container dstack-pytorch-1  Started
```

If you see `libspdm expects LKCA but found stubs!`, you booted the
wrong image (probably `0.6.0`). Re-pull `0.6.1+`.

If you see `Failed to get app keys from KMS ... Failed to decode
attestation`, the KMS at `:13001` is older than your guest image.
Update the KMS to a version built from a recent `meta-dstack-cloud`
commit.

---

## 5. Verify the GPU is in Confidential Compute Mode

### 5.1 Status from the host

```bash
dstack-cloud status
# Status: RUNNING
# External IP: <ip>
# Gateway URLs:
#   App URL:      https://<app_id>-8090.tdxlab.dstack.org:13004/
#   Instance URL: https://<instance_id>-8090.tdxlab.dstack.org:13004/
```

### 5.2 SSH into the guest (if §3.4 was applied)

Append to `~/.ssh/config`:

```
Host gpu-hello
    User root
    ProxyCommand openssl s_client -verify_quiet -quiet -connect <app_id>-22.tdxlab.dstack.org:13004
```

```bash
ssh gpu-hello
```

### 5.3 `nvidia-smi`

From inside the guest:

```bash
nvidia-smi
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
# |...
# |   0  NVIDIA H100 80GB HBM3          On  |   00000000:04:00.0 Off |                    0 |
# | N/A   31C    P0            119W /  700W |      23MiB /  81559MiB |      0%      Default |
```

```bash
nvidia-smi conf-compute -f
# CC status: ON
nvidia-smi conf-compute -grs
# Confidential Compute GPUs Ready state: ready
```

### 5.4 PyTorch matmul

The container in §3.3 prints its result to the dstack log:

```bash
dstack-cloud logs -n 30000 | grep -A4 "CUDA available"
# CUDA available: True
# Device: NVIDIA H100 80GB HBM3
# Capability: (9, 0)
# 100x 4096^3 matmul: 0.361s -> 38074 GFLOPs
```

You should see ~38 TFLOPs FP32 on a single 4096³ matmul. Anything
significantly lower likely means the GPU is fallback-running on host
CPU (i.e. CUDA never initialized; double-check `nvidia-smi` from
within the container too).

---

## 6. Resource Cleanup

```bash
dstack-cloud remove          # deletes the VM + auto-attached disks
gsutil rm gs://<bucket>/dstack-gpu-hello-shared.tar.gz  # optional
```

Spot instances on `a3-highgpu-1g` are billed by the second; remove
them when you're not actively testing.

---

## Appendix A — Building the OS image yourself

If you need a kernel/driver change beyond what `0.6.1` ships, build a
custom image:

```bash
git clone --recurse-submodules https://github.com/Phala-Network/meta-dstack-cloud
cd meta-dstack-cloud
./scripts/setup-host.sh                       # installs yocto host deps
source dev-setup
bitbake -c cleansstate virtual/kernel mc:nvidia:nvidia
FLAVORS=nvidia make dist                      # ~30–60 min on a fast box
ls bb-build/dist/
# dstack-cloud-nvidia-0.6.1.tar.gz
# dstack-cloud-nvidia-0.6.1-uki.tar.gz
# dstack-cloud-nvidia-0.6.1/disk.raw + auth_hash.txt
```

Drop the resulting `dstack-cloud-nvidia-<version>/` directory into
your `image_search_paths` and reference it from `app.json`'s
`os_image`.

The minimum kernel-config delta required for H100 CC is one line —
`CONFIG_CRYPTO_ECDSA=y` in
`meta-dstack/recipes-kernel/linux/files/6.18/defconfig` — and the
nvidia recipe must be rebuilt against that kernel so the
`USE_LKCA` macro takes effect in `nvidia.ko`.

---

## Appendix B — Troubleshooting cheat sheet

| Symptom (in `dstack-cloud logs`)                                            | Likely cause                                                                | Fix |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----|
| `Failed to decode attestation from cert extension`                          | KMS at `:13001` is older than your guest image                              | Update KMS to a current `meta-dstack-cloud` build |
| `libspdm expects LKCA but found stubs!`                                     | Booted `dstack-cloud-nvidia-0.6.0` (or older); kernel missing `CRYPTO_ECDSA`  | Use `0.6.1+` image |
| `gpumgrCheckRmFirmwarePolicy: Disabling GSP offload -- GPU not supported`   | Previous failed init latched the GPU; needs PCIe FLR                        | `nvidia-smi -r`, or reboot the instance |
| `ZONE_RESOURCE_POOL_EXHAUSTED`                                              | Spot H100 capacity gone in this zone                                        | Try another H100 zone from §2.2 |
| `Invalid value for field 'resource.name': '<...>.<...>'`                    | Instance name contains `.`                                                  | Rename the project; GCP image names disallow dots |
| KMS connection times out                                                    | Local config still has the wrong port (`:12001`/`:12002`)                  | Verify `~/.config/dstack-cloud/config.json` ports match §2.4 |

---

## References

- [Phala-Network/meta-dstack-cloud](https://github.com/Phala-Network/meta-dstack-cloud) — TEE image source
- [Phala-Network/dstack-cloud-deployment-guide](https://github.com/Phala-Network/dstack-cloud-deployment-guide) — KMS-on-GCP deployment (companion guide)
- [GCP — Create a Confidential VM instance with GPU](https://docs.cloud.google.com/confidential-computing/confidential-vm/docs/create-a-confidential-vm-instance-with-gpu)
- [NVIDIA — Confidential Computing on H100](https://docs.nvidia.com/confidential-computing/index.html)
