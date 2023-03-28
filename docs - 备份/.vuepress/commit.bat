@echo off
git add . 
git commit -m "Via script" 
call proxy.bat git push -u origin master
::https://www.cnblogs.com/yayin/p/13691239.html
pause