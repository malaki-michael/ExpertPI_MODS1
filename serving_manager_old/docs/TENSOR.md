How to start torchserve service on tensor computer:

 1. Connect to tensor via remote desktop
 2. Open powershell as administrator
 3. By clicking uparrow, navigate to the script (it will cycle through history), which opens all necessary ports. This script is also present on the desktop under file startup.ps1.
 4. Open terminal app.
 5. Open new tab with Ubuntu 22.04.
 6. When Ubuntu 22.04 is opened, disconnect from remote desktop and test connection. Open cmd.
 7. Desired response: "status": "healthy" - ```curl http://172.16.2.86:8080/ping```
 8. Desired response: list of models - ```curl http://172.16.2.86:8081/models```