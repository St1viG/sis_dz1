import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal
# %%
start_t = -8
d_t = 0.01
end_t = 8
t = np.arange(start_t, end_t, d_t)
x = pow(2, t / 2) * ((t >= -1) ^ (t >= 5))

plt.figure()
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Originalni signal')
# plt.show()

t0 = 12
skaliranje = 2

y1 = np.concatenate((x, np.zeros(int(t0 / d_t))))
t1 = np.concatenate((np.arange(t[0] - t0, t[0], d_t), t))
plt.figure()
plt.plot(t1, y1)
plt.xlabel('t1')
plt.ylabel('y1(t)')
plt.title('Pomeren signal')
# plt.show()

# 2. inverzija
y2 = y1[::-1]
t2 = -t1[::-1]
plt.figure()
plt.plot(t2, y2)
plt.xlabel('t2')
plt.ylabel('y2(t)')
plt.title('Invertovan signal')
# plt.show()

# 3. skaliranje
y3 = y2[1:-1:skaliranje]
t3 = t2[1:-1:skaliranje] / skaliranje
nt = np.linspace(t3[0], t3[-1], len(t3) * skaliranje)  # Interpolation time points
ny = np.interp(nt, t3, y3)
plt.figure()
plt.plot(nt, ny)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Finalni signal')
# plt.show()

plt.figure()
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('Originalni signal')
# plt.show()

t0 = -8
skaliranje = 1 / 2

# 1. pomeranje
y1 = np.concatenate((x, np.zeros(int(abs(t0) / d_t))))  # kasnjenje
# potrebno je prosiriti vremensku osu (ili skratiti x) zbog iscrtavanja
t1 = np.concatenate((t, np.arange(t[-1] + d_t, t[-1] - t0 + d_t, d_t)))
plt.figure()
plt.plot(t1, y1)
plt.xlabel('t1')
plt.ylabel('y1(t)')
plt.title('Pomeren signal')
# plt.show()

# 2. sklaliranje
N = len(y1)
t2 = np.arange(t1[0] / skaliranje, t1[-1] / skaliranje, d_t)
y2 = np.interp(t2, t1 / skaliranje, y1)
plt.figure()
plt.plot(t2, y2)
plt.xlabel('t2')
plt.ylabel('y2(t)')
plt.title('Skaliran signal')
# plt.show()

# %%
# kod realnih signala, poput govornog, nema smisla razmatrati negativno vreme odnosno treba ih posmatrati od t=0 nadalje


# snimanje i analiza zvucnog signala
samplerate = 8000  # ucestanost odabiranja
duration = 3  # trajanje snimka

# myrecording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
# sd.wait()  # ceka dok se zavrsi snimanje
# wavfile.write('sekvenca.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom

samplerate, data = wavfile.read('./sek.wav')

dt = 1 / samplerate
t = np.arange(0, dt * len(data), dt)
chanel1 = data  # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.plot(t, chanel1)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal')
# plt.show()

# c1
skaliranje = 0.5
t1 = np.arange(t[0] / skaliranje, t[-1] / skaliranje, d_t)
y2 = np.interp(t1, t / skaliranje, chanel1)
plt.figure()
plt.plot(t1, y2)
plt.xlabel('t2')
plt.ylabel('y2(t)')
plt.title('Skaliran Audio signal')
# plt.show()
print("Playing S(n/2) with frequency:8kHz")
# sd.play(y2, samplerate)
sd.wait()

# c2
skaliranje = 2
y2 = chanel1[1:-1:skaliranje]
t1 = t[1:-1:skaliranje] / skaliranje
plt.figure()
plt.plot(t1, y2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Skaliran signal')
# plt.show()
print("Playing S(2n) with frequency:8kHz")
# sd.play(y2, samplerate)
sd.wait()

# c3
print('Playing original audio file')
# sd.play(chanel1, samplerate)

sd.wait()
print('Playing modified file with frequency:4kHz')
# sd.play(chanel1, 4000)

sd.wait()
print('Playing modified file with frequency:16kHz')
# sd.play(chanel1, 16000)

# c4

y2 = chanel1[::-1]
t2 = -t1[::-1]
t2 = np.arange(0, dt * len(chanel1), dt)
plt.figure()
plt.plot(t2, y2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Invertovan signal')
# plt.show()

# sd.play(y2, 8000)
sd.wait()

# c5
# print("Recording")
# myrecording = sd.rec(int(4 * samplerate), samplerate=samplerate, channels=1)
# sd.wait()  # ceka dok se zavrsi snimanje
# print("Recording ended")
# wavfile.write('palindrom.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom


samplerate, data = wavfile.read('./palindrom.wav')
# sd.play(data, samplerate)
sd.wait()
dt = 1 / samplerate
t = np.arange(0, dt * len(data), dt)
chanel1 = data  # chanel1=data[:,1] ako ima dva kanala
plt.figure()
plt.plot(t, chanel1)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Palindrom - Audio signal')
# plt.show()

# inverzija palindromnog signala

y2 = chanel1[::-1]
t2 = -t1[::-1]
t2 = np.arange(0, dt * len(chanel1), dt)
plt.figure()
plt.plot(t2, y2)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Invertovan signal')
# plt.show()

# sd.play(y2, samplerate)
sd.wait()


print()



# 2. zad
#2.a
start_t = -8
d_t = 0.01
end_t = 8
t = np.arange(start_t, end_t, d_t)
x = pow(2, t / 2) * ((t >= -1) ^ (t >= 5))
td = np.arange(start_t ,end_t ,d_t)
delta = np.zeros_like(td)
delta[len(td)// 2 - (2*100)]  = 1/4
delta[len(td)// 2 + (2*100)]  =  1/4
y = np.convolve(delta, x, mode='same')
plt.plot(td, y)
plt.title("konvolucija")
plt.xlabel("t")
plt.ylabel("conv(t)")
# plt.show()

#2.b


# samplerate=44000
# print("Recording")
# myrecording = sd.rec(int(3 * samplerate), samplerate=44000, channels=1)
# sd.wait()  # ceka dok se zavrsi snimanje
# print("Recording ended")
# wavfile.write('words.wav', samplerate, myrecording)  # cuva se fajl po odgovarajucim imenom

samplerate_sekvenca, sekvenca = wavfile.read('./words.wav')
# sekvenca=sekvenca[:,1]
dt = 1 / samplerate_sekvenca
t = np.arange(0, dt * len(sekvenca), dt)
plt.figure()
plt.plot(t, sekvenca)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Audio signal')
# plt.show()

odzivi =['./Impulsni odzivi/OutdoorStadium.wav','./Impulsni odzivi/CastleThunder.wav','./Impulsni odzivi/GiantCave.wav','./Impulsni odzivi/Hangar.wav','./Impulsni odzivi/RoomPool.wav']



samplerate_impulsni, impulsni_odziv = wavfile.read(odzivi[4])
impulsni_odziv = impulsni_odziv / max(np.absolute(impulsni_odziv))
dt_i = 1 / samplerate_impulsni
# t_i = np.arange(0, dt * len(impulsni_odziv), dt_i)
t_i = np.linspace(0, (len(impulsni_odziv) - 1) * dt_i, len(impulsni_odziv))
plt.figure()
plt.plot(t_i, impulsni_odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Impulsni odziv')
sd.play(sekvenca,44000)
sd.wait()
sekvenca = sekvenca / np.max(np.abs(data))

odziv = np.convolve(sekvenca, impulsni_odziv)
odziv = odziv / max(np.absolute(odziv))
dt_o = 1 / samplerate_impulsni
# t_o = np.arange(0, dt * len(odziv), dt_o)
t_o = np.arange(len(odziv)) * dt_o

plt.figure()
plt.plot(t_o, odziv)
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Konvolucija')
# plt.show()
sd.play(impulsni_odziv,samplerate_impulsni)
sd.wait()
sd.play(odziv, samplerate_impulsni)
sd.wait()































# energija signala je veoma bitna karakteristika signala, a definisana je kao Ex=sum[k=-inf,inf](x^2(k))

# govorni signal spada u one signale koji su dinamicni i brzo menjaju svoje
# karakeristike, zbog toga se njegove karakteristike cesto posmatraju u
# takozvanim prozorima, tj.kratkim intervalima vremena, od oko 20-30ms. Upravo
# ova karakteristika moze da pomogne oko tzv. segmentacije govora, tj.
# izdvajanja reci koristeci adekvatno izabrane pragove. Realizacija takvog
# jednostavnog algoritma ce biti predmet domaceg zadatka
# wl = int(samplerate * 20e-3)  # prozor od 20 milisekundi
# E = np.zeros(len(chanel1))  # energija
# for i in range(wl, len(chanel1)):
#     E[i] = sum(chanel1[i - wl:i] ** 2)  # suma kvadrata prethodnih wl odbiraka
# plt.figure()
# plt.plot(t, E)
# plt.xlabel('t')
# plt.ylabel('E(t)')
# plt.title('Energija audio signala')
# plt.show()
# # %%
# sd.play(data, samplerate)
# # %%
#
# samplerate_sekvenca, sekvenca = wavfile.read('./sekv.wav')
# # sekvenca=sekvenca[:,1]
# dt = 1 / samplerate_sekvenca
# t = np.arange(0, dt * len(sekvenca), dt)
# plt.figure()
# plt.plot(t, sekvenca)
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Audio signal')
# plt.show()
#
# samplerate_impulsni, impulsni_odziv = wavfile.read('./impulses_airwindows_RoomMedium.wav')
# impulsni_odziv = impulsni_odziv / max(np.absolute(impulsni_odziv))
# dt_i = 1 / samplerate_impulsni
# t_i = np.arange(0, dt * len(impulsni_odziv), dt_i)
# plt.figure()
# plt.plot(t_i, impulsni_odziv)
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Impulsni odziv')
# plt.show()
#
# odziv = np.convolve(sekvenca, impulsni_odziv)
# odziv = odziv / max(np.absolute(odziv))
# dt_o = 1 / samplerate_impulsni
# t_o = np.arange(0, dt * len(odziv), dt_o)
# plt.figure()
# plt.plot(t_o, odziv)
# plt.xlabel('t')
# plt.ylabel('y(t)')
# plt.title('Konvolucija')
# plt.show()
#
# # %%
# sd.play(sekvenca, samplerate_sekvenca)
#
# # %%
# sd.play(impulsni_odziv, samplerate_impulsni)
#
# # %%
# sd.play(odziv, samplerate_sekvenca)
#
#
# # %%
# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
#
#
# img = mpimg.imread('./peppers.png')
# img = rgb2gray(img)
# plt.figure()
# imgplot = plt.imshow(img, cmap=plt.get_cmap('gray'))
# plt.show()
#
# # Blur
# K_blur = np.ones((3, 3)) * 1 / 9
# img_blur = signal.convolve2d(img, K_blur, mode='same').clip(0, 1)
# plt.figure()
# imgplot = plt.imshow(img_blur, cmap=plt.get_cmap('gray'))
# plt.show()
#
# # Sharp
#
# K_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# img_sharp = signal.convolve2d(img, K_sharp, mode='same').clip(0, 1)
# plt.figure()
# imgplot = plt.imshow(img_sharp, cmap=plt.get_cmap('gray'))
# plt.show()
#
# # %%
# # Furijeovi redovi: razmatra se signal koji je unipolarna povorka cetvrtki periode 2s,
# # izracunati su koeficijenti Furijeovog reda koji iznose
# t = np.arange(-2, 2, 0.01)
# x = 1 * ((t >= -2) * (t < -1)) + 1 * ((t >= 0) * (t < 1))
# plt.figure()
# plt.plot(t, x)
# plt.show()
# # %% skicirati amplitudski i fazni linijski spektar
#
# N = 50
# pi = np.pi
# w0 = pi
# k = np.arange(-N, N + 1)
# ak = 1 / (1j * 2 * k * pi) * (1 - np.exp(-1j * k * pi))
# ak[N] = 0.5  # a0=0.5 (srednja vrednost signala - DC komponenta)
#
# plt.figure()
# plt.stem(k, np.absolute(ak))
# plt.xlabel('k')
# plt.title('Amplitudski linijski spektar')
# plt.show()
#
# plt.figure()
# plt.stem(k, np.angle(ak))
# plt.xlabel('k')
# plt.title('Fazni linijski spektar')
# plt.show()
#
# # %%  napraviti rekonstrukciju originalnog signala koristeci N=50 harmonika
#
# t = np.arange(-2, 2, 0.01)
# x = 1 * ((t >= -2) * (t < -1)) + 1 * ((t >= 0) * (t < 1))
# N = 50
# x1 = np.zeros(len(t))
# for i in range(len(t)):
#     x1[i] = np.absolute(ak[N])
#     for k in range(1, N + 1):
#         x1[i] = x1[i] + 2 * np.absolute(ak[k + N]) * np.cos((k) * w0 * t[i] + np.angle(ak[k + N]))
# plt.figure()
# plt.plot(t, x)
# plt.plot(t, x1)
# plt.show()
#
# # %%  napraviti rekonstrukciju originalnog signala koristeci N=10 harmonika
#
# N = 10
# pi = np.pi
# w0 = pi
# k = np.arange(-N, N + 1)
# ak = 1 / (1j * 2 * k * pi) * (1 - np.exp(-1j * k * pi))
# ak[N] = 0.5  # a0=0.5 (srednja vrednost signala - DC komponenta)
# t = np.arange(-2, 2, 0.01)
# x = 1 * ((t >= -2) * (t < -1)) + 1 * ((t >= 0) * (t < 1))
# x2 = np.zeros(len(t))
# for i in range(len(t)):
#     x2[i] = np.absolute(ak[N])
#     for k in range(1, N + 1):
#         x2[i] = x2[i] + 2 * np.absolute(ak[k + N]) * np.cos((k) * w0 * t[i] + np.angle(ak[k + N]))
# plt.figure()
# plt.plot(t, x)
# plt.plot(t, x2)
# plt.show()
# # %%
# plt.figure()
# plt.plot(t, x)
# plt.plot(t, x1)
# plt.plot(t, x2)
# plt.show()
# # %%
# # ako zelimo da preskocimo ak[0], pa da ga umetnemo na kraju
# # k = np.arange(-N,0)
# # ak_neg = 1/(1j*2*k*pi)*(1-np.exp(-1j*k*pi))
# # k = np.arange(1,N+1)
# # ak_pos = 1/(1j*2*k*pi)*(1-np.exp(-1j*k*pi))
# # a0=0.5
# # k=range(-N,N+1)
# # ak=np.concatenate((ak_neg,np.array([a0])))
# # ak=np.concatenate((ak,ak_pos))
