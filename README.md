# Nemesis_AEGIS 🤖

**Artificial Intelligence algorithm** that performs the aptamers folding using a GAN Network, it is a Generative adversarial network, divided in two neuronal networks: the Generative and the Discriminative. CNN nets are used.
<p align="center"><img src="https://github.com/Zildj1an/SELEX/blob/master/OT_Robot/img/molecule2.gif" alt="" width="400"/></p>

_Generated aptamer_

This algorithm is part of the project of the UCM_Madrid Igem Team, all the information about the entire project can be found in our <a href = "https://2019.igem.org/Team:MADRID_UCM/Landing">website</a>.

The modeling part of the project is **SELEX Process with Artificial Intelligence** 👨‍🔧. And it includes:

a) The robotic automatization of biotechnological protocols

b) The usage of Artifical Intelligence

c) A search engine

d) Many, many custom hardware

e) And a website design

You can find more information about the modeling <a href = "https://github.com/Zildj1an/SELEX">here</a>. 

# Installation and Usage ❓

In the folder DB_Creation, execute (if you use Mac or Linux), for terminal:
```
$ python DB_Creation_Terminal.py
```
You can also speed things up with threads with bash run_several.sh, but be aware that you would need to modify the absolute path to the Rosetta software.

In the folder GAN_Network, from terminal:
```
$ python Main.py
```
Other files that could be used if the judge wants:
- 3D_structure_creation_in_pdb.py Creates only one PDB for the DDBB (testing puposes)
- GAN_Network.py (Same thing as GAN_Network folder but in one file)
