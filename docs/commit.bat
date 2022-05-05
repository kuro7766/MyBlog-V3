@echo off
git add . 
git commit -m "Via script" 
call proxy.bat git push -u origin master
pause
pause