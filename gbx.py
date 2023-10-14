from pygbx import Gbx, GbxType

g = Gbx(r"C:\Users\Daniel\Documents\TrackMania\Tracks\Challenges\My Challenges\Level 3a.Challenge.Gbx")
#g = Gbx(r"C:\Users\Daniel\Documents\TrackMania\Tracks\Challenges\My Challenges\AStrangePotato's track.Challenge.Gbx")
challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
if not challenges:
    quit()

challenge = challenges[0]

n = 30
a = []
for i in range(n):
    a.append([" "] * n)
    

o=0
for block in challenge.blocks:
    if "StadiumRoadMain" in block.name:
        p = block.position
        a[p.x-o][p.z-o] = "#"
    if block.name == "StadiumRoadMainStartLine" or block.name == "StadiumRoadMainFinishLine":
        print(block)
        p = block.position
        a[n-p.x][p.z-o] = "%"
    if block.name =="StadiumCircuitBase":
        p = block.position
        a[n-p.x][p.z-o] = "#"

    #print(block.name)

        
for n in a:

    print(*n)
