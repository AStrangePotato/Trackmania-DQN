from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import pickle


roadBlocks = [(528, 528), (528, 512), (528, 496), (528, 480), (512, 480), (496, 480), (480, 480), (480, 464), (480, 448), (480, 432), (480, 416), (480, 400), (464, 400), (448, 400), (432, 400), (432, 416), (432, 432), (432, 448), (416, 448), (400, 448), (400, 464), (400, 480), (400, 496), (416, 496), (432, 496), (432, 512), (432, 528), (432, 544), (432, 560), (448, 560), (464, 560), (464, 544), (464, 528), (480, 528), (496, 528), (496, 544), (496, 560), (512, 560), (528, 560), (544, 560), (560, 560), (576, 560), (576, 576), (576, 592), (576, 608), (592, 608), (608, 608), (624, 608), (624, 592), (624, 576), (640, 576), (656, 576), (656, 560), (656, 544), (656, 528), (640, 528), (624, 528), (608, 528), (592, 528), (576, 528), (576, 512), (576, 496), (576, 480), (576, 464), (576, 448), (560, 448), (544, 448), (528, 448), (512, 448), (512, 432), (512, 416), (528, 416), (544, 416), (544, 400), (544, 384), (560, 384), (576, 384), (592, 384), (608, 384), (608, 400), (608, 416), (608, 432), (608, 448), (608, 464), (608, 480), (624, 480), (640, 480), (640, 464), (640, 448), (656, 448), (672, 448), (672, 432), (672, 416), (672, 400), (656, 400), (640, 400), (640, 384), (640, 368), (656, 368), (672, 368), (688, 368), (688, 352), (688, 336), (704, 336), (720, 336), (720, 352), (720, 368), (720, 384), (720, 400), (720, 416), (720, 432), (736, 432), (752, 432), (752, 448), (752, 464), (752, 480), (736, 480), (720, 480), (704, 480), (688, 480), (688, 496), (688, 512), (704, 512), (720, 512), (720, 528), (720, 544), (720, 560), (704, 560), (688, 560), (688, 576), (688, 592), (688, 608), (688, 624), (688, 640), (688, 656), (688, 672), (672, 672), (656, 672), (656, 656), (656, 640), (640, 640), (624, 640), (608, 640), (592, 640), (576, 640), (560, 640), (544, 640), (528, 640), (528, 624), (528, 608), (528, 592), (512, 592), (496, 592), (480, 592), (464, 592), (448, 592), (432, 592), (416, 592), (400, 592), (384, 592), (368, 592), (368, 576), (368, 560), (368, 544), (368, 528), (368, 512), (368, 496), (368, 480), (368, 464), (368, 448), (368, 432), (368, 416), (368, 400), (368, 384), (368, 368), (384, 368), (400, 368), (416, 368), (432, 368), (448, 368), (464, 368), (480, 368), (496, 368), (496, 352), (496, 336), (512, 336), (528, 336), (544, 336), (560, 336), (576, 336), (592, 336), (608, 336), (624, 336), (640, 336), (656, 336), (656, 320), (656, 304), (656, 288), (656, 272), (656, 256), (656, 240), (656, 224), (656, 208), (656, 192), (656, 176), (656, 160), (640, 160), (624, 160), (608, 160), (592, 160), (576, 160), (560, 160), (544, 160), (528, 160), (512, 160), (496, 160), (496, 176), (496, 192), (496, 208), (496, 224), (496, 240), (496, 256), (496, 272), (512, 272), (528, 272), (528, 256), (528, 240), (528, 224), (528, 208), (528, 192), (544, 192), (560, 192), (560, 208), (560, 224), (560, 240), (576, 240), (592, 240), (592, 224), (592, 208), (608, 208), (624, 208), (624, 224), (624, 240), (624, 256), (624, 272), (624, 288), (624, 304), (608, 304), (592, 304), (576, 304), (560, 304), (544, 304), (528, 304), (512, 304), (496, 304), (480, 304), (464, 304), (464, 288), (464, 272), (464, 256), (464, 240), (464, 224), (464, 208), (464, 192), (448, 192), (432, 192), (416, 192), (416, 208), (416, 224), (400, 224), (384, 224), (368, 224), (368, 240), (368, 256), (384, 256), (400, 256), (416, 256), (432, 256), (432, 272), (432, 288), (432, 304), (432, 320), (432, 336), (416, 336), (400, 336), (384, 336), (368, 336), (352, 336), (336, 336), (336, 352), (336, 368), (336, 384), (320, 384), (304, 384), (288, 384), (288, 368), (288, 352), (288, 336), (288, 320), (288, 304), (304, 304), (320, 304), (336, 304), (336, 288), (336, 272), (336, 256), (336, 240), (336, 224), (320, 224), (304, 224), (288, 224), (272, 224), (272, 240), (272, 256), (272, 272), (256, 272), (240, 272), (240, 256), (240, 240), (240, 224), (240, 208), (240, 192), (240, 176), (256, 176), (272, 176), (288, 176), (304, 176), (320, 176), (336, 176), (336, 160), (336, 144), (352, 144), (368, 144), (384, 144), (400, 144), (416, 144), (432, 144), (448, 144), (464, 144), (464, 128), (464, 112), (480, 112), (496, 112), (512, 112), (528, 112), (544, 112), (560, 112), (576, 112), (592, 112), (608, 112), (624, 112), (640, 112), (656, 112), (672, 112), (688, 112), (688, 128), (688, 144), (688, 160), (688, 176), (688, 192), (688, 208), (688, 224), (688, 240), (688, 256), (688, 272), (688, 288), (688, 304), (704, 304), (720, 304), (736, 304), (752, 304), (768, 304), (784, 304), (800, 304), (816, 304), (816, 288), (816, 272), (816, 256), (816, 240), (800, 240), (784, 240), (784, 256), (784, 272), (768, 272), (752, 272), (736, 272), (720, 272), (720, 256), (720, 240), (720, 224), (720, 208), (720, 192), (736, 192), (752, 192), (752, 176), (752, 160), (768, 160), (784, 160), (784, 176), (784, 192), (784, 208), (800, 208), (816, 208), (832, 208), (848, 208), (848, 224), (848, 240), (848, 256), (848, 272), (848, 288), (848, 304), (848, 320), (848, 336), (848, 352), (848, 368), (832, 368), (816, 368), (816, 352), (816, 336), (800, 336), (784, 336), (784, 352), (784, 368), (784, 384), (784, 400), (784, 416), (800, 416), (816, 416), (832, 416), (832, 432), (832, 448), (832, 464), (816, 464), (800, 464), (784, 464), (784, 480), (784, 496), (800, 496), (816, 496), (832, 496), (848, 496), (864, 496), (880, 496), (880, 480), (880, 464), (880, 448), (880, 432), (880, 416), (880, 400), (880, 384), (880, 368), (880, 352), (880, 336), (880, 320), (880, 304), (880, 288), (896, 288), (912, 288), (912, 304), (912, 320), (912, 336), (912, 352), (912, 368), (912, 384), (912, 400), (912, 416), (912, 432), (912, 448), (912, 464), (912, 480), (912, 496), (912, 512), (912, 528), (912, 544), (896, 544), (880, 544), (864, 544), (864, 560), (864, 576), (880, 576), (896, 576), (896, 592), (896, 608), (896, 624), (880, 624), (864, 624), (848, 624), (848, 640), (848, 656), (864, 656), (880, 656), (896, 656), (912, 656), (912, 672), (912, 688), (896, 688), (880, 688), (864, 688), (848, 688), (832, 688), (816, 688), (816, 672), (816, 656), (816, 640), (816, 624), (816, 608), (816, 592), (816, 576), (816, 560), (816, 544), (800, 544), (784, 544), (768, 544), (768, 560), (768, 576), (768, 592), (768, 608), (768, 624), (752, 624), (736, 624), (736, 640), (736, 656), (736, 672), (752, 672), (768, 672), (768, 688), (768, 704), (768, 720), (768, 736), (784, 736), (800, 736), (816, 736), (832, 736), (848, 736), (864, 736), (880, 736), (896, 736), (896, 752), (896, 768), (896, 784), (896, 800), (880, 800), (864, 800), (848, 800), (832, 800), (832, 784), (832, 768), (816, 768), (800, 768), (784, 768), (784, 784), (784, 800), (784, 816), (784, 832), (784, 848), (784, 864), (784, 880), (768, 880), (752, 880), (736, 880), (736, 896), (736, 912), (720, 912), (704, 912), (704, 896), (704, 880), (704, 864), (704, 848), (704, 832), (720, 832), (736, 832), (752, 832), (752, 816), (752, 800), (752, 784), (736, 784), (720, 784), (704, 784), (704, 768), (704, 752), (704, 736), (688, 736), (672, 736), (656, 736), (640, 736), (624, 736), (624, 720), (624, 704), (624, 688), (608, 688), (592, 688), (576, 688), (560, 688), (544, 688), (528, 688), (512, 688), (496, 688), (480, 688), (480, 672), (480, 656), (464, 656), (448, 656), (432, 656), (416, 656)]

states = []

last_capture_time = 0
class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        

    def on_run_step(self, iface: TMInterface, _time: int):
        global last_capture_time
        
        if _time > 0 and _time % 50 == 0:
            state = iface.get_simulation_state()
            pos = state.position
            for i in range(60):
                for j in range(60):
                    if i*16-8 < pos[0] < i*16+8:
                        if j*16-8 < pos[2] < j*16+8:
                            blockCenter = (i*16, j*16)
                            if blockCenter not in roadBlocks:
                                roadBlocks.append(blockCenter)
                                print(blockCenter)

            if _time - last_capture_time >= 2000:
                last_capture_time = _time
                states.append(state)
                print(f"Captured state #{len(states)} at {_time} ms")
                
if __name__ == "__main__":
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)



def save():
    with open("trainingStates.sim", "wb") as f:
        pickle.dump(states, f)
        print(f"Saved {len(states)} states.")