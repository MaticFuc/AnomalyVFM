from typing import Any, List, NamedTuple

# Industrial
from datasets.mvtec_ad import MVTecTestDataset
from datasets.visa import VisATestDataset
from datasets.mpdd import MPDDTestDataset
from datasets.btad import BTADTestDataset
from datasets.real_iad import RealIADTestDataset
from datasets.ksdd import KSDDTestDataset
from datasets.ksdd2 import KSDD2TestDataset
from datasets.dagm import DAGMTestDataset
from datasets.dtd import DTDTestDataset

# Medical Image level
from datasets.headct import HeadCTTestDataset
from datasets.br35h import Br35hTestDataset
from datasets.brainmri import BrainMRITestDataset

# Medical Pixel level
from datasets.isic import ISICTestDataset
from datasets.kvasir import KvasirTestDataset
from datasets.endo import EndoTestDataset
from datasets.tn3k import ThyroTestDataset
from datasets.cvc_clinicdb import CVC_ClinicDBTestDataset
from datasets.cvc_colondb import CVC_ColonDBTestDataset

# Datasets outside the original paper
# Industrial
from datasets.real_iad_variety import RealIADVarietyTestDataset
from datasets.goods_ad import GoodsADTestDataset
from datasets.rsdd import RSDDTestDataset

# 2.5D Datasets (currently only RGB)
from datasets.mvtec_3d import MVTec3DTestDataset
from datasets.eyecandies import EyecandiesTestDataset
from datasets.real_iad_d3 import RealIAD3DTestDataset


DATASET_ROOT="./data/"


class DatasetProperties(NamedTuple):
    name: str
    path: str
    init_class: Any
    class_names: List[str]


DATASET_RESOURCES = [
    # Datasets from the original paper
    # Industrial
    DatasetProperties( "mvtec_ad", f"{DATASET_ROOT}/mvtec", MVTecTestDataset, [ "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper", ], ),
    DatasetProperties( "visa", f"{DATASET_ROOT}/visa", VisATestDataset, [ "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum", ], ),
    DatasetProperties( "real_iad", f"{DATASET_ROOT}/REAL_IAD", RealIADTestDataset, [ "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood", "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug", "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape", "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1", "u_block", "usb", "usb_adaptor", "vcpill", "wooden_beads", "woodstick", "zipper", ], ),
    DatasetProperties( "mpdd", f"{DATASET_ROOT}/MPDD", MPDDTestDataset, [ "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes", ], ),
    DatasetProperties( "btad", f"{DATASET_ROOT}/BTech_Dataset_transformed", BTADTestDataset, ["01", "02", "03"], ),
    DatasetProperties( "ksdd", f"{DATASET_ROOT}/KSDD", KSDDTestDataset, [ "electrical commutators", ], ),
    DatasetProperties( "ksdd2", f"{DATASET_ROOT}/KSDD2", KSDD2TestDataset, ["metal_plates"], ),
    DatasetProperties( "dagm", f"{DATASET_ROOT}/DAGM_split/DAGM_KaggleUpload", DAGMTestDataset, [ "Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10", ], ),
    DatasetProperties( "dtd", f"{DATASET_ROOT}/DTD-Synthetic", DTDTestDataset, [ "Woven_001", "Woven_127", "Woven_104", "Stratified_154", "Blotchy_099", "Woven_068", "Woven_125", "Marbled_078", "Perforated_037", "Mesh_114", "Fibrous_183", "Matted_069", ], ),
    # Medical Image level
    DatasetProperties( "headct", f"{DATASET_ROOT}/HeadCT", HeadCTTestDataset, ["head_ct"], ),
    DatasetProperties( "brainmri", f"{DATASET_ROOT}/BrainMRI", BrainMRITestDataset, ["brain"], ),
    DatasetProperties( "br35h", f"{DATASET_ROOT}/Br35H", Br35hTestDataset, ["brain"], ),
    # Medical Pixel level
    DatasetProperties( "isic", f"{DATASET_ROOT}/isbi", ISICTestDataset, ["skin"] ),
    DatasetProperties( "cvc_colondb", f"{DATASET_ROOT}/polyp/CVC-ColonDB", CVC_ColonDBTestDataset, ["polyp"], ),
    DatasetProperties( "cvc_clinicdb", f"{DATASET_ROOT}/polyp/CVC-ClinicDB", CVC_ClinicDBTestDataset, ["polyp"], ),
    DatasetProperties( "kvasir", f"{DATASET_ROOT}/polyp/Kvasir", KvasirTestDataset, ["polyp"], ),
    DatasetProperties( "endo", f"{DATASET_ROOT}/EndoTect_2020_Segmentation_Test_Dataset", EndoTestDataset, ["polyp"], ),
    DatasetProperties( "thyro", f"{DATASET_ROOT}/thyro", ThyroTestDataset, ["tn3k"], ),

    # Datasets beyond the original paper
    # Industrial
    DatasetProperties("real_iad_variety", f"{DATASET_ROOT}/REAL_IAD_VARIETY/realiadvariety_raw/", RealIADVarietyTestDataset, ["3_adapter","3pin_aviation_connector","4_wire_stepping_motor","access_card","accurate_detection_switch","aircraft_model_head","angled_toggle_switch","audio_jack_socket","bag_buckle","ball_pin","balun_transformer","battery","battery_holder_connector","battery_socket_connector","bend_connector","blade_switch","blue_light_switch","bluetooth_module","boost_converter_module","bread_model","brooch_clasp_accessory","button_battery_holder","button_motor","button_switch","car_door_lock_switch","ceramic_fuse","ceramic_wave_filter","charging_port","chip_inductor","circuit_breaker","circular_aviation_connector","common_mode_choke","common_mode_filter","connector","connector_housing_female","console_switch","crimp_st_cable_mount_box","dc_jack","dc_power_connector","detection_switch","D_sub_connector","duckbill_circuit_breaker","DVD_switch","earphone_audio_unit","effect_transistor","electronic_watch_movement","ethernet_connector","ferrite_bead","ffc_connector_plug","flow_control_valve","flower_copper_shape","flower_velvet_fabric","fork_crimp_terminal","fuse_cover","fuse_holder","gear","gear_motor","green_ring_filter","hairdryer_switch","hall_effect_sensor","headphone_jack_female","headphone_jack_socket","hex_plug","humidity_sensor","ingot_buckle","insect_metal_parts","inverter_connector","jam_jar_model","joystick_switch","kfc_push_key_switch","knob_cap","laser_diode","lattice_block_plug","LED_indicator","lego_pin_connector_plate","lego_propeller","lego_reel","lego_technical_gear","lego_turbine","lighting_connector","lilypad_led","limit_switch","lithium_battery_plug","littel_fuse","little_cow_model","lock","long_zipper","meteor_hammer_arrowhead","miniature_laser_module","miniature_lifting_motor","miniature_motor","miniature_stepper_motor","mobile_charging_connector","model_steering_module","monitor_socket","motor_bracket","motor_gear_reducer","motor_plug","mouse_socket","multi_function_switch","nylon_ball_head","optical_fiber_outlet","pencil_sharpener","pinboard_connector","pitch_connector","PLCC_socket","pneumatic_elbow","pot_core","potentiometer","power_bank_module","power_inductor","power_jack","power_strip_socket","pulse_transformer","purple_clay_pot","push_button_switch","push_in_terminal","recorder_switch","rectangular_connector_accessories","red_terminal","retaining_ring","rheostat","rotary_position_sensor","round_twist_switch","self_lock_switch","shelf_support_module","side_press_switch","side_release_buckle","silicon_cell_sensor","sim_card_reader","single_pole_potentiometer","single_switch","slide_switch","slipper_model","small_leaf","smd_receiver_module","solderless_adapter","spherical_airstone","spring_antenna","square_terminal","steering_T_head","steering_wheel","suction_cup","telephone_spring_switch","tension_snap_release_clip","thumbtack","thyristor","toy_tire","traceless_hair_clip","travel_switch","travel_switch_green","tubular_key_switch","vacuum_switch","vehicle_harness_conductor","vertical_adjustable_resistor","vibration_motor","volume_potentiometer","VR_joystick","wireless_receiver_module"]),
    DatasetProperties("goods_ad", f"{DATASET_ROOT}/GoodsAD", GoodsADTestDataset, ['cigarette_box', 'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package']),
    DatasetProperties("rsdd", f"{DATASET_ROOT}/RSDD", RSDDTestDataset, ['Metal15']),
    
    # Industrial 3D
    DatasetProperties("mvtec_3d", f"{DATASET_ROOT}/mvtec3d", MVTec3DTestDataset, ["cable_gland", "bagel", "cookie", "carrot", "dowel", "foam", "peach", "potato", "tire", "rope",]),
    DatasetProperties("eyecandies", f"{DATASET_ROOT}/eyecandies/Eyecandies/", EyecandiesTestDataset, ["CandyCane", "ChocolateCookie", "ChocolatePraline", "GummyBear", "HazelnutTruffle", "LicoriceSandwich", "Lollipop", "Marshmallow", "PeppermintCandy"]),
    DatasetProperties("real_iad_3d", f"{DATASET_ROOT}/REAL_IAD_3D", RealIAD3DTestDataset, ["audio_jack_socket", "connector_housing_female", "dc_power_connector", "ferrite_bead", "fuse_holder", "humidity_sensor", "lattice_block_plug", "lego_propeller", "miniature_lifting_motor", "purple_clay_pot", "common_mode_filter", "crimp_st_cable_mount_box", "ethernet_connector", "fork_crimp_terminal", "headphone_jack_socket", "knob_cap", "lego_pin_connector_plate", "limit_switch", "power_jack", "telephone_spring_switch"]),
    
]
