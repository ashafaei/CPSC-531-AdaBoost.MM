function H = wl_brief( train_data, bits )
%WL_BRIEF Summary of this function goes here
%   Detailed explanation goes here

blurred      = train_data.blurred;

m = size(train_data.blurred, 3);

responses = false(m, bits);

patterns = randi([1 32], [bits 4]);

for i=1:bits,
   responses(:, i) =  squeeze(blurred(patterns(i, 1), patterns(i, 2), :) > blurred(patterns(i, 3), patterns(i, 4), :));
end

H = struct();
H.responses = responses;
H.patterns  = patterns;
H.labels    = train_data.labels;
H.bits      = bits;
H.m         = m;

end

