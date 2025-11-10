%quantisiert zu einem fixen Wertebereich [−1, +1] und floor rundet ab
function q= quantize(s,num_bits)
 num_levels= 2^num_bits;
 q= floor(s * num_levels/2) * (2/num_levels); %floor rundet ab
 end