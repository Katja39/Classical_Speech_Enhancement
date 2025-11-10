%einfache, lineare quantisierung 
function q= quantopt(s,num_bits)
 num_levels= 2^num_bits;
 scalingFactor= (max(s)-min(s)) / num_levels;
 q= round(s / scalingFactor) * scalingFactor; %round, einfaches runden
 end
