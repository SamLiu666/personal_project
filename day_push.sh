#!/bin/bash
git status
git add -A
echo -n " Please Enter Your Push Record ->"
read -r record
echo -n record
git commit -m $record
git push test master

echo "done"
