#!/bin/bash

thejar="Plato.jar"
if [ $# == 1 ] 
then
    thejar=$1
fi

read -p "User: " user
read -p "Host: " host

ssh -f $user@$host -L5000:cypriot.stanford.edu:5000 -N
ssh -f $user@$host -L3306:playfair.stanford.edu:3306 -N

java -cp $thejar openproof.plato.Plato $user $host

