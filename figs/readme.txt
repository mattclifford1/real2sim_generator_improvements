edit using inkscape

tex text addon:
# remove snap install of inkscape

snap remove inkscape
sudo add-apt-repository ppa:inkscape.dev/stable
sudo apt update
sudo apt install inkscape



# install pdflatex
sudo apt-get install texlive-full  

# download tex text from github link on here
https://textext.github.io/textext/install/linux.html#for-systems-with-inkscape-installed-from-a-package-manager
# unzip and install
python3 setup.py
