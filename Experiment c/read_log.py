f = open('log.txt','r')
p_l = []
for i in range(1,101,1):
    line = f.readline()
    psnr = float(line[-11:-4])
    p_l.append(psnr)
max_psnr = max(p_l)
print(max_psnr)
print(p_l.index(max_psnr) + 1)
    