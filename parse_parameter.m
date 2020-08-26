% Asks for keyboard input, replacing void by a default parameter
function [P] = parse_parameter(prompt, default)
P = input(prompt);
if (isempty(P))
    P = default;
end
end
