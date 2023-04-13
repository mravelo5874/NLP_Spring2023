''' example tasks '''
# python run.py --model m2m --task example
# python run.py --model m2m --task mono-sim --words small --l0 english
# python run.py --model m2m --task duo-sim --words small --l0 french --l1 spanish
# python run.py --model m2m --task duo-sim --words pereira --l0 japanese --l1 korean

''' 4/13/2023 examples with duo-sim '''
# C:\Users\Marco\Documents\GitHub\NLP_Spring2023\Final Project>python run.py --model m2m --task duo-sim --l0 english --l1 french
# english words:  ['small', 'short', 'child', 'wife', 'mother', 'construction', 'capitalism', 'capitalist', 'communism', 'father']
# french words:  ['petite', 'courte', 'enfant', 'épouse', 'mère', 'construction', 'le capitalisme', 'capitaliste', 'communisme', 'père']
# Computing similarity between languages: english and french. This may take some time...
# The similarity between english and french is: 0.539631

# C:\Users\Marco\Documents\GitHub\NLP_Spring2023\Final Project>python run.py --model m2m --task duo-sim --l0 french --l1 spanish
# french words:  ['petite', 'courte', 'enfant', 'épouse', 'mère', 'construction', 'le capitalisme', 'capitaliste', 'communisme', 'père']
# spanish words:  ['pequeño', 'corto', 'niño', 'mujer', 'madre', 'construcción', 'el capitalismo', 'capitalista', 'el comunismo', 'padre']
# Computing similarity between languages: french and spanish. This may take some time...
# The similarity between french and spanish is: 0.600580

''' 4/13/2023 examples with duo-sim '''
# C:\Users\Marco\Documents\GitHub\NLP_Spring2023\Final Project>python run.py --model mbart --task duo-sim --words pereira --l0 english --l1 german
# Gathering word lists for duo-sim calculation...
# device set to:  cpu
# created list file:  ./word_lists/german_pereira_mbart.json
# Computing similarity between languages: 'english' and 'german'. This may take some time...
# device set to:  cpu
# device set to:  cpu
# The similarity between 'english' and 'german' is: 0.210328
# elapsed time:  270.09

# C:\Users\Marco\Documents\GitHub\NLP_Spring2023\Final Project>python run.py --model m2m --task duo-sim --words pereira --l0 japanese --l1 korean
# Gathering word lists for duo-sim calculation...
# device set to:  cpu
# created list file:  ./word_lists/japanese_pereira_m2m.json
# created list file:  ./word_lists/korean_pereira_m2m.json
# Computing similarity between languages: 'japanese' and 'korean'. This may take some time...
# device set to:  cpu
# device set to:  cpu
# The similarity between 'japanese' and 'korean' using 'm2m' is: 0.229778
# elapsed time:  844.28

''' example sentences: '''
# 'The sun rose over the mountains, casting a golden glow across the valley.'
# 'The city\'s heartbeat echoed through the night, a symphony of sirens, footsteps, and distant laughter, as the streets pulsed with life and energy, relentless and unforgiving.'
# 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'
# 'The theoretical framework adopted in this research drew upon established theories in the field of social sciences, providing a solid conceptual foundation for the study and guiding the formulation of research questions and hypotheses.'
# 'The scent of roses wafted through the air, mingling with the salty tang of the sea, as the sun dipped below the horizon, casting the world into twilight.'