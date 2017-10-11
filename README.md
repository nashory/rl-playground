# rl-playground
my playground for testing RL-algorithms



## Prerequisites

__(recommended) Create virtualenv with:  __
~~~
(we assume you already installed python-pyode using "sudo apt-get install python-pyode" command.)
virtualenv --system-site-packages --python=python2.7 venv
pip install -r requirements.txt
~~~

+ PyOpenGL 
`pip install PyOpenGL PyOpenGL_accelerate`

+ Open Dynamics Engine (pyODE)
`sudo apt-get install python-pyode`

+ cgkit
ref : https://codeyarns.com/2013/08/24/how-to-build-and-install-cgkit-from-source-on-ubuntu/
~~~
sudo apt-get install scons python-pygame
sudo wget https://sourceforge.net/projects/cgkit/files/cgkit/cgkit-2.0.0/
cd supportlib
scons
cd ..
sudo python setup.py install
~~~



## Usage

## Author  
Minchul Shin, [@nashory](https://githubcom/nashory)





