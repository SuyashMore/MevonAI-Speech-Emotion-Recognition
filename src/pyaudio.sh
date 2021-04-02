sudo apt-get purge --remove python3-pyaudio
cd portaudio/
./configure
make
make install
pip3 install pyaudio

