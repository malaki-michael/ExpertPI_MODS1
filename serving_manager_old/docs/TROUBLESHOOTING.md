# Inference + TEM Plugins problem solving guide
Explore application has integrated checker for inference server and TEM plugins. Sometimes, the application may not work as expected and this guide is intended to help you troubleshoot the problem.


## First aid guide

There are usually three possible ways how to solve the problem:
  1. Locate **ml_startup.bat** file on the Desktop of the Explore computer. Please note that for F2, global Tensor computer is used and you need to restart it there. Run the .bat file and then proceed to [Check if docker is running](#check-docker-running).
  2. Open Powershell.exe and type:
   ```powershell
   wsl.exe --shutdown
   ```
   Then open Ubuntu application from the start menu and run. Proceed to [Check if docker is running](#check-docker-running).

  3. Restart the computer. Cmd window with nohup process [this link](https://en.wikipedia.org/wiki/Nohup) should be started automatically if not (report to Branislav Hesko), start ubuntu terminal from start menu. Proceed to [Check if docker is running](#check-docker-running).

## Check docker running
Open Ubuntu terminal window and run:
```bash
docker stats
```
There should always be **plugins_tem_plugins_1** and **torchserve_torchserve_1** running.

## Inspection for docker logs
Open Ubuntu terminal window and run:
```bash
docker logs <container_id> --follow -n 1000
```
or
```bash
docker logs <container_id>
```
Where `<container_id>` is the id/name of the container you want to inspect. You can find the container id by running `docker stats` command.

## Check application logs
To check application logs, open Ubuntu terminal window and run:
```bash
cd ~/data
```
Then run:
```bash
explorer.exe .
```
You can see images, stitchings and other things logged inside. HTML logs are located in a subfolder **logs**. These logs contain information about a complete runs for stitching and deprecession coefficients tunning.


## How to turn off data logging from inference
  1. To speed up inference, there is a possibility to turn off data logging from inference.
  2. Open Ubuntu terminal, type **explorer.exe .**.
  3. Navigate to *docker* folder.
  4. Open **.env** file with any text editor. Backup the previous version. Change these variables in the following way:
     ```
     PLUGIN_HOST=""
     PLUGIN_PORT=""
     ```
  5. Save and close the .env file.
  6. Open Ubuntu terminal `cd docker`.
  7. Run `docker-compose down`.
  8. Run `docker-compose up -d`.
  9. Proceed to check if the server is running.

To restore the previous version of the .env file, copy the backup file to the .env file and run `docker-compose down` followed by `docker-compose up -d`.


# How to solve inference server problems
Healthy server should garant that loaded models are fully functional. Sometimes, the server may not respond to particular requests and more steps to fix this behavior are needed.
Currently, there are these status messages:

## What server status messages mean

 * **Healthy** - everything should be fine, inference should be as expected.
 * **Unavailable** - server is not available, it is not responding to requests. Usually this should not represent a problem, because the server may not be reachable due to VPN connection or other network settings.
 * **Unhealthy** - server is not healthy, it is responding to requests, but the inference is not working as expected. This is the status that should be fixed.
 * **Partially unhealthy** - server is not healthy, it is responding to requests, inference may be running, but some models are not working as expected. This is the status that should be fixed.

## How to fix unhealthy server

### Check server logs
If you have access to the computer where the server is running, you can check the logs. We are using docker to run the server, so you can first find the container by running `docker ps` command. Then you can check the logs by running `docker logs <container_id>`.

### Check server status using web interface
**Web interface is available at `http://172.16.2.86:5001/TorchServe_Checker` or `http://172.19.1.16:5001/TorchServe_Checker`. You can check the status there.**


### Check server status
If you have access to the computer where the server is running, you can check the server status by running `curl localhost:8080/ping` command. You may replace `localhost` by the corresponding IPv4 address, like `172.16.2.86`. The output should be something like this:
```
{
  "status": "Healthy",
}
```
**Always make sure Ubuntu / Linux terminal is running on the target machine!!**

### Fix unhealthy(partially unhealthy) server
These steps are suggested to fix unhealthy server:
 * Restart the computer where the server runs. This should fix the problem in most cases.
 * If the server is still unhealthy, you can try to restart the server by running `docker restart <container_id>` command. You can find the container id by running `docker ps` command.
 * Contact `branislav.hesko@tescan.com`.

### Fix unavailable server
These steps are suggested to fix unavailable server:
 * Check if the server is running locally by following the steps above.
 * Make sure the PC server is reachable under vpn by using `ping IP_ADDRESS` command.
 * Run powershell script to allow ports to be forwarded.
  ```powershell
  $remoteport = bash.exe -c "ip a show eth0 | grep 'inet '"
 $found = $remoteport -match '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}';

 if ($found) {
   $remoteport = $matches[0];
 }
 else {
   Write-Output "IP address could not be found";
   exit;
 }

 $ports = @(5000, 7444, 7443, 8080, 8081, 8082);

 for ($i = 0; $i -lt $ports.length; $i++) {
   $port = $ports[$i];
   Invoke-Expression "netsh interface portproxy delete v4tov4 listenport=$port";
   Invoke-Expression "netsh advfirewall firewall delete rule name=$port";

   Invoke-Expression "netsh interface portproxy add v4tov4 listenport=$port connectport=$port connectaddress=$remoteport";
   Invoke-Expression "netsh advfirewall firewall add rule name=$port dir=in action=allow protocol=TCP localport=$port";
 }

 Invoke-Expression "netsh interface portproxy show v4tov4";
  ```
 * Contact IT department.
 * Contact `branislav.hesko@tescan.com`.
