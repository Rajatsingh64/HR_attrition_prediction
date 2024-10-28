from setuptools import find_packages , setup 
import os , sys 
from typing import List 

requirement_file= "requirements.txt"

with open (requirement_file)  as file:
    requirement_list = file.readlines() #reading requirements.txt line by line and store name inside list
#removing  "/n"  or space from require_list 
requirement_list = [require_name.replace("\n" , "") for require_name in requirement_list if "\n" in require_name]  
if "_e ." in requirement_list:
    requirement_list.remove("_e .")


#making Project package

setup(
     name="src" , 
     author="Rajat Singh" , 
     author_email="rajat.k.singh64@gmail.com" , 
     version="0.1" , 
     packages=find_packages()  , 
     install_requires = requirement_list


 
)
 