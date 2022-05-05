@echo off
git add . 
git commit -m "Via script" 
proxy.bat git push -u origin master
pause
pause