# Instructions for WSL2 inference server.

1. Install WSL2 on the computer according to https://pureinfotech.com/install-windows-subsystem-linux-2-windows-10/
2. Pick an user, usually use computer name and password for him.
3. For more smooth experience install **Terminal** (Microsoft publisher) from Windows Store app.
4. Update and upgrade the system:
   ```
   sudo apt update && sudo apt upgrade
   ```
5. Install systemd, docker.
   ```
   sudo apt install docker.io systemd docker-compose-v2
   ```
6. Allow automatic systemd run by adding the following snippet into */etc/wsl.conf*. Use *sudo nano /etc/wsl.conf* to open and edit the file:
   ```
   [boot]
   systemd=true
   ```
7. Restart WSL by entering *wsl --shutdown* into powershell. Start new ubuntu terminal session.
8. Enable docker and add user to docker group:
   ```
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```
9. restart WSL again and enter *docker stats*. You should see all running docker containers.
10. Install nvidia docker support from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html. This involves the following snippet:
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    sudo docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
    ```
    At the end you should see similar window:
    ```
    Fri Jan 27 07:44:03 2023
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.05    Driver Version: 528.24       CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
    |  0%   38C    P8    15W / 215W |    246MiB /  8192MiB |      8%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A        24      G   /Xwayland                       N/A      |
    +-----------------------------------------------------------------------------+
    ```
11. Copy docker folder from this repository into WSL filesystem into user folder /home/USER_NAME/. You should see Linux filesystem inside explorer.exe on the windows.
12. Go into docker folder in WSL in your terminal window. Run the command:
    ```
    docker build -f Dockerfile -t stem_torchserve .
    ```
13. Copy your models into /home/user/models (~/models) folder.
14. Run: *docker compose up -d* to run the inference server.
15. Open cmd.exe under windows and type: *curl http://127.0.0.1:8080/ping* - You should see "healthy" response.
16. Again in cmd.exe type command: *curl http://127.0.0.1:8081/models/TEMRegistration* - replace TEMRegistration by whatever model you are using. You should see model information and a single worker in workers list. Response should look like:
    ```
    [
        {
            "modelName": "TEMRegistration",
            "modelVersion": "1.0",
            "modelUrl": "TEMRegistration.mar",
            "runtime": "python",
            "minWorkers": 1,
            "maxWorkers": 1,
            "batchSize": 1,
            "maxBatchDelay": 100,
            "loadedAtStartup": true,
            "workers": [
            {
                "id": "9000",
                "startTime": "2023-01-27T07:53:42.945Z",
                "status": "READY",
                "memoryUsage": 0,
                "pid": 39,
                "gpu": true,
                "gpuUsage": "gpuId::0 utilization.gpu [%]::8 % utilization.memory [%]::2 % memory.used [MiB]::885 MiB"
            }
            ]
        }
    ]
    ```
17. Open powershell as administrator.
18. ```New-NetFirewallRule -DisplayName "WSL2 Port Bridge" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8081, 8080, 7443, 7444, 8082```
19. For each port in [7443, 7444, 8080, 8081, 8082], ```netsh interface portproxy add v4tov4 listenport=8081 listenaddress=0.0.0.0 connectport=8081 connectaddress=192.168.161.177```, IP address is where WSL2 responds to curl command. Usually is written in "ip address" command in ubuntu.


# Instruction to run TEM-Plugins
TEM Plugins is a python fastapi server providing various functionality from stitching, quality metrics to logging and watchdog options. It runs as a docker container under wsl2. It has access to the gpu, but there is currently no plugin using it. To install it please follow this tutorial:

1. ```git clone git@github.com:tescan-orsay-holding/TEM-Plugins.git```
2. Copy the TEM-Plugins folder onto WSL2, no preference for exact location.
3. ```cd TEM-Plugins```
4. ```docker build -f docker/Dockerfile -t tem_plugins .```
5. If the previous step goes wrong, probably the internet connection cannot connect to tescan pypi server.
6. In this case, manually download pypi packages needed (see Dockerfile) from pypi and put them into TEM-Plugins. Download url: ```gitlab.tescan.com/api/v4/projects/84/packages/pypi/simple```. Username and password are written in Dockerfile.
7. Run local docker installation via: ```docker build -f docker/Dockerfile-no-pypi -t tem_plugins .```.
8. ```docker compose -f docker/docker-compose.yml up -d```.
9. If docker compose is not available try: ```sudo apt install docker-compose-v2```.
10. ```docker compose --version``` should be at least 1.29 for GPU support.
11. Go to http://localhost:5000/docs, if Plugins are running you should see OpenAPI simple endpoint page.
12. Check file ```.env```, there are environment variables for Plugins. If you want to change them, you need to restart the docker container. Specifically, **SAVE_IMAGES** and **DEBUG_LOGGER** should be used only on development machines, not on production. They are allowing to save images and log debug messages into files. This is not recommended for production. Make sure that mapping of volumes is correct and path exists. It can be modified in docker-compose.yml file.


# Instructions for automatic startup of docker containers in WSL2
1. WSL2 already has systemd installed. Docker will be started by systemd automatically when WSL2 is started. WSL2 cannot be currently started without user login. This is a problem for automatic startup of docker containers. To solve this problem, WSL2 startup should be added into Windows Startup folder.
2. Open Windows Explorer and type ```shell:startup``` into address bar. This will open startup folder.
3. Copy install/startup.bat into this folder.
4. After restart of Windows, WSL2 should start automatically and docker containers should be started automatically as well. You should see a terminal window running. This window prevents Windows from turning off WSL2.
5. ```nvidia-smi.exe``` should show models running and https://localhost:5000/docs should be accessible.
6. Usually, WSL2 is not forwarded to the internet. To forward it, powershell script is included in install/startup.ps1. Please run powershell as administator and run this script. It will forward ports 8080, 8081, 7443, 7444, 8082 to/from WSL2. This is needed for TEM-Plugins and TorchServe to work correctly.
7. Explore server will be afterwards callable as IP_V4_ETH_ADDRESS:PORT. Ip address can be found by checking properties of ethernet adapter in Windows.
8. If scripts cannot be executed on the machine: Go to Windows Settings / Security / For developers and check "Trust local scripts". This will allow scripts to be executed on the machine.
9. Finally create a scheduler task, where startup.ps1 script will be run each time user is logged on. Please note, this step might not be needed.