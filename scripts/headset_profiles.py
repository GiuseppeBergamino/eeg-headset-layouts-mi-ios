"""Structured headset profiles and electrode-layout utilities."""
from dataclasses import dataclass                   
from typing import List, Optional, Tuple, Dict      


# 10-20 old names -> modern 10-20 names
# TP9/TP10 could be approximated to the closest available 10–10 sites in Cho2017 (P9/P10).
LEGACY_MAP: Dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
   # "TP9": "P9",
   # "TP10": "P10"
}

def normalize_channel_names(ch: str) -> str:
    #Map legacy 10–20 labels (T3/T4/T5/T6) to modern equivalents (T7/T8/P7/P8)
    return LEGACY_MAP.get(ch, ch)


# Dataclass definition
@dataclass
class HeadsetProfile:
    name: str                                       # headset name (e.g., EMOTIV_EPOC_X)
    channels: List[str]                             # EEG channel names in MNE format
    
    company_name: Optional[str] = None              # company name
    label: Optional[str] = None                     # headset label (e.g., Emotiv EPOC X 14 ch)
    form_factor: Optional[str] = None               # form factor (e.g., Headphones, Headband, Helmet, Cap)
    sampling_rate: Optional[List[float]] = None     # sampling rate in Hz
    bit_depth: Optional[int] = None                 # bit depth 
    reference: Optional[List[str]] = None           # reference-channel names in MNE format
    montage: Optional[str] = None                   # montage (e.g., 10-20, 10-10)
    bandwidth: Optional[Tuple[float, float]] = None # in Hz, if onboard preprocessing is documented (DC block, anti-aliasing, etc)
    weight: Optional[float] = None                  # weight in grams
    size: Optional[str] = None                      # dimensions in cm or adjustable range
    battery_life: Optional[float] = None            # in hours
    charging_time: Optional[float] = None           # in hours
    connectivity: Optional[str] = None              # e.g., Wired USB, Wireless 2.4 GHz, WiFi, Bluetooth, BLE
    wireless_range: Optional[float] = None          # in meters 
    sensor_type: Optional[str] = None               # electrode type (e.g., Dry, Wet saline, Wet gel)
    sensor_material: Optional[str] = None           # electrode material (e.g., Ag/AgCl, Au, Conductive polymer)
    extra_sensors: Optional[str] = None             # extra sensors (e.g., EOG, EMG, ECG, IMU)
    raw_data_access: Optional[str] = None           # e.g., real-time, offline, or both
    raw_data_availability: Optional[str] = None     # e.g., SDK, APIs, paid app, cloud
    product_link: Optional[str] = None              # official product link
    price: Optional[float] = None                   # price in EUR
    donning_time: Optional[float] = None            # declared donning time in minutes
    notes: Optional[str] = None                     # optional notes


    @property
    def nr_channels(self) -> int:
        return len(self.channels)

    # Create the dataset-specific channel mask for this headset
    def make_mask(
            self,                           # headset profile instance
            available_channels: List[str],  # available EEG channels in the dataset
            verbose: bool = True            # verbose by default
        ) -> List[str]: 
            
            # map legacy names when needed
            mask_channels = [normalize_channel_names(ch) for ch in self.channels]
            available_norm = {normalize_channel_names(ch) for ch in available_channels}

            matching = [ch for ch in mask_channels if ch in available_norm]
            missing = [ch for ch in mask_channels if ch not in available_norm]

            if verbose:
                print("Headset name:", self.name)
                print("TOT channels:", len(self.channels))
                print("OK channels:", len(matching), "/", len(self.channels))
                if len(missing) > 0:
                    print(len(missing), "Missing channels:", missing)
            
            return matching 
    

# --------------------------- Custom profiles --------------------------- #
# Order for coverage reporting (front to back)
# COVERAGE_ORDER = ["Fp","AF","F","FC","FT","T","TP","C","CP","P","PO","O","Iz","A"]
IDEAL_SENSORIMOTOR = HeadsetProfile(
    name = "IDEAL_SENSORIMOTOR",
    label = "Ideal sensorimotor headset 15 ch",
    channels = [
         'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
         'C3',   'C1',  'Cz',  'C2', 'C4',
         'CP3', 'CP1', 'CPz', 'CP2', 'CP4'
    ],
    montage= "10-10"
    )

# ----------------Cho2017---------------- #
FULL_CHO = HeadsetProfile(
     name = "FULL_CHO",
     label = "Headset dataset Cho 2017",
     channels = [ 
          'Fp1', 'AF7', 'AF3',  'F1',  'F3',  'F5',  'F7', 'FT7', 
          'FC5', 'FC3', 'FC1',  'C1',  'C3',  'C5',  'T7', 'TP7', 
          'CP5', 'CP3', 'CP1',  'P1',  'P3',  'P5',  'P7',  'P9', 
          'PO7', 'PO3',  'O1',  'Iz',  'Oz', 'POz',  'Pz', 'CPz', 
          'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz',  'Fz',  'F2',  'F4', 
          'F6',   'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',  'Cz', 
          'C2',   'C4',  'C6',  'T8', 'TP8', 'CP6', 'CP4', 'CP2', 
          'P2',   'P4',  'P6',  'P8', 'P10', 'PO8', 'PO4',  'O2'
          ],
    company_name = "Biosemi ActiveTwo",
    form_factor = "Cap",
    sampling_rate = [512.0],
    bit_depth = None,
    reference = None,
    montage = "10-10",
    sensor_material = "Ag/AgCl"
)

# ----------------EMOTIV HEADSETS---------------- #
EMOTIV_FLEX = HeadsetProfile(
    name = "EMOTIV_FLEX",
    channels = [
                    'Fp1', 'Fp2',
          'F7',   'F3', 'Fz',  'F4',      'F8',
     'FT9',  'FC5',  'FC1', 'FC2', 'FC6', 'FT10',
          'T7', 'C3',   'Cz',  'C4',  'T8',
     'TP9', 'CP5',  'CP1', 'CP2', 'CP6', 'TP10',
          'P7',   'P3', 'Pz',  'P4',  'P8',
                  'O1', 'Oz', 'O2'
    ],
    company_name = "Emotiv",
    label = "Emotiv Flex 2 Saline/Gel 32 ch",
    form_factor = "Cap",
    sampling_rate = [256.0],
    bit_depth = 16,
    reference = ['AFz', 'FCz'], 
    montage = "extended 10-20",
    bandwidth = (0.2, 45.0),
    weight= 170.0,
    size = "3 cap sizes: S (54 cm), M (56 cm), L (58 cm)",
    battery_life = 6.0,
    charging_time = 2.0,
    connectivity = "Bluetooth 5.2",
    wireless_range = None,
    sensor_type = "Wet saline or wet gel",
    sensor_material = "Ag/AgCl",
    extra_sensors = "ICM-20948 9 axis IMU",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid App",
    price = 1900.0,
    product_link = "https://www.emotiv.com/products/flex-saline"
)

EMOTIV_EPOCH_X = HeadsetProfile(
    name = "EMOTIV_EPOCH_X",
    channels = [
                 'AF3', 'AF4',
        'F7',     'F3',  'F4',      'F8',
              'FC5',          'FC6',
        'T7',                       'T8',
        'P7',                       'P8',
                  'O1',  'O2'
    ],
    company_name = "Emotiv",
    label = "Emotiv EPOC X 14 channels",
    form_factor = "Helmet",
    sampling_rate = [128.0, 256.0],
    bit_depth = 14,
    reference = ['CMS', 'DRL'], #proprietary reference
    montage = "extended 10-20",
    bandwidth = (0.16, 43.0),
    weight= 170.0,
    size = "9 x 15 x 15 cm, flexible",
    battery_life = 6.0,
    charging_time = 2.0,
    connectivity = "Bluetooth LE, 2.4 GHz, USB",
    wireless_range = None,
    sensor_type = "Wet saline",
    sensor_material = "Ag/AgCl + saline soaked felt pads",
    extra_sensors = "ICM-20948 9 axis IMU",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid App",
    price = 1200.0,
    product_link = "https://www.emotiv.com/products/epoc-x"
)

EMOTIV_INSIGHT = HeadsetProfile(
    name = "EMOTIV_INSIGHT",
    channels = [
           'AF3', 'AF4',
        'T7',          'T8',
                'Pz'
    ],
    company_name = "Emotiv",
    label = "Emotiv Insight 5 ch",
    form_factor = "Helmet",
    sampling_rate = [128.0],
    bit_depth = 16,
    reference = ['CMS', 'DRL'], #left and right mastoids
    montage = "extended 10-20",
    bandwidth = (0.5, 45.0),
    weight = None,
    size = None,
    battery_life = 20.0,
    charging_time= 2.0,
    connectivity = "Bluetooth LE, 2.4 GHz,",
    wireless_range = None,
    sensor_type = "Wet saline",
    sensor_material = "Semi dry polymer",
    extra_sensors = "ICM-20948 9 axis IMU",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid App",
    price = 600.0,
    product_link = "https://www.emotiv.com/products/insight"
)

EMOTIV_MN8 = HeadsetProfile(
    name = "EMOTIV_MN8",
    channels = [
        'T7',  'T8',
    ],
    company_name = "Emotiv",
    label = "Bluetooth headphones 2 ch",
    form_factor = "Headphones",
    sampling_rate = [128.0],
    bit_depth = 14,
    reference = None,
    montage = "extended 10-20",
    bandwidth = (0.5, 45.0),
    weight = None,
    size = None,
    battery_life = 6.0,
    charging_time = 2.0,
    connectivity = "Bluetooth LE",
    wireless_range = None,
    sensor_type = "Dry",
    sensor_material = "Conductive elastomer",
    extra_sensors = "Microphone, ICM-20948 6 axis IMU",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid App",
    price = 480.0,
    product_link = "https://www.emotiv.com/products/mn8"
)

# ----------------ULTRACORTEX HEADSETS---------------- #
ULTRACORTEX_16_MILIMB = HeadsetProfile(
    name = "ULTRACORTEX_16_MILIMB",
    channels = [ 
               'AF3', 'AF4', 
         'F7',  'F3',  'F4',  'F8',
         'T7',  'C3',  'C4',  'T8',
         'P7',  'P3',  'P4',  'P8',
                'O1',  'O2'
    ],
    company_name = "OpenBCI",
    label = "Ultracortex Mark IV 16 ch",
    form_factor = "Helmet",
    sampling_rate = [125.0],
    bit_depth = 24,
    reference = ['T7', 'T8'],
    montage = "10-20",
    bandwidth = (5.0, 50.0),
    weight = None,
    size = None,
    battery_life = None,
    charging_time= None,
    connectivity = "Wireless Bluetooth",
    wireless_range = None,
    sensor_type = "Dry",
    sensor_material = "Conductive polymer",
    extra_sensors = "9-axis IMU (MPU-9250)",
    raw_data_access = "Real-time",
    raw_data_availability = "Open Source SDK",
    price = 518.0,
    product_link = "https://docs.openbci.com/AddOns/Headwear/MarkIV/",
)

ULTRACORTEX_16 = HeadsetProfile(
    name = "ULTRACORTEX_16",
    channels = [
                  'Fp1', 'Fp2',
            'F7',  'F3',  'F4',  'F8',
            'T7',  'C3',  'C4',  'T8',
            'P7',  'P3',  'P4',  'P8',
                   'O1',  'O2'
   ],
    company_name = "OpenBCI",
    label = "Ultracortex Mark IV 16 ch",
    form_factor = "Helmet",
    sampling_rate = [125.0],
    bit_depth = 24,
    reference = ['T7', 'T8'],
    montage = "10-20",
    bandwidth = (5.0, 50.0),
    weight= None,
    size= None,
    battery_life= None,
    charging_time= None,
    connectivity = "Wireless Bluetooth",
    wireless_range= None,
    sensor_type = "Dry",
    sensor_material = "Conductive polymer",
    extra_sensors = "9-axis IMU (MPU-9250)",
    raw_data_access = "Real-time",
    raw_data_availability = "Open Source SDK",
    price = 600.0,
    product_link = "https://docs.openbci.com/AddOns/Headwear/MarkIV/",
)

ULTRACORTEX_8 = HeadsetProfile(
    name = "ULTRACORTEX_8",
    channels = [
                 'Fp1', 'Fp2',
               'C3',       'C4',
            'P7',             'P8',
                   'O1',  'O2'
    ],
    company_name = "OpenBCI",
    label = "Ultracortex Mark IV 8 ch",
    form_factor = "Helmet",
    sampling_rate = [125.0],
    bit_depth = 24,
    reference = ['T7', 'T8'],
    montage = "10-20",
    bandwidth = (5.0, 50.0),
    weight= None,
    size= None,
    battery_life= None,
    charging_time= None,
    connectivity = "Wireless Bluetooth",
    wireless_range= None,
    sensor_type = "Dry",
    sensor_material = "Conductive polymer",
    extra_sensors = "9-axis IMU (MPU-9250)",
    raw_data_access = "Real-time",
    raw_data_availability = "Open Source SDK",
    price = 450.0,
    product_link = "https://docs.openbci.com/AddOns/Headwear/MarkIV/",
)

#------G.Tec HEADSETS------#
GTEC_UNICORN_HYBRID= HeadsetProfile(
     name = "GTEC_UNICORN_HYBRID",
     channels = [
                'Fz',
          'C3', 'Cz', 'C4',
                'Pz',
          'PO7',     'PO8',
                'Oz'
     ],
     label = "g.Tec Unicorn Hybrid 8 ch",
     company_name = "g.Tec",
     form_factor = "Cap",
     sampling_rate = [250.0],
     bit_depth = 24,
     reference= ['earlobes'],
     montage = "10-20",
     bandwidth = (0.0, 100.0),
     weight = 56.0,
     size = "Adjustable 54-60 cm",
     battery_life= 3.0,
     charging_time= 2.0,
     connectivity = "Cabled USB, Bluetooth",
     wireless_range= 10.0,
     sensor_type = "Wet gel or dry",
     sensor_material = None,
     extra_sensors = "6 axis IMU",
     raw_data_access= "Real-time",
     raw_data_availability= "Paid SDK, APIs      ",
     price = 1200.0,
     product_link = "https://www.gtec.at/product/unicorn-hybrid-black/"
)

#------------ Bitbrain HEADSETS ------------#
BITBRAIN_DIADEM = HeadsetProfile(
        name = "BITBRAIN_DIADEM",
        channels = [
                    'Fp1', 'Fp2',
             'AF7',              'AF8',
                 'F3',       'F4',
                 'P3',       'P4',
             'PO7',              'PO8',
                     'O1',  'O2'
        ],
        company_name = "Bitbrain",
        label = "Bitbrain Diadem 12 ch",
        form_factor= "Helmet",
        sampling_rate = [256.0],
        bit_depth = 24,
        reference = ['A1', 'Fpz'],
        montage = "10-20",
        bandwidth = (0.0, 40.0),
        weight = 300.0,
        size = "Adjustable 53-61 cm",
        battery_life = 8.0,
        charging_time = 3.0,
        connectivity = "Bluetooth",
        wireless_range= 10.0,
        sensor_type = "Dry",
        sensor_material = None,
        extra_sensors = "9 axis IMU",
        raw_data_access= "Real-time, On-board storage (8GB)",
        raw_data_availability= "SDK",
        price = 14200.0,
        product_link = "https://www.bitbrain.com/neurotechnology-products/dry-eeg/diadem"
)
BITBRAIN_AIR = HeadsetProfile(
        name = "BITBRAIN_AIR",
        channels = [
                   'Fp1', 'Fp2',
             'AF7',             'AF8',
             'PO7',             'PO8',
                    'O1',  'O2'
        ],
        label = "Bitbrain AIR 8 ch",
        company_name="Bitbrain",
        form_factor= "Helmet",
        sampling_rate = [256.0],
        bit_depth = 24,
        reference = ['A1', 'Fpz'],
        montage = "10-20",
        bandwidth = (0.0, 40.0),
        weight = 210.0,
        size = "Adjustable 53-61 cm",
        battery_life = 8.0,
        charging_time = 3.0,
        connectivity = "Bluetooth",
        wireless_range= 10.0,
        sensor_type = "Dry",
        sensor_material = None,
        extra_sensors = "9 axis IMU",
        raw_data_access= "Real-time, On-board storage (8GB)",
        raw_data_availability= "SDK",
        price = 6100.0,
        product_link = "https://www.bitbrain.com/neurotechnology-products/dry-eeg/air"
)
BITBRAIN_HERO = HeadsetProfile(
        name = "BITBRAIN_HERO",
        channels = [
             'FC3', 'FCz', 'FC4',
             'C3',  'Cz',  'C4',
             'CP3', 'CPz', 'CP4',
        ],
        label = "Bitbrain Hero 9 ch",
        company_name="Bitbrain",
        form_factor= "Helmet",
        sampling_rate = [256.0],
        bit_depth = 24,
        reference = ['A1', 'A2'],
        montage = "10-20",
        bandwidth = (0.0, 40.0),
        weight= 250.0,
        size = "Head breadth 13.5, 16.5 cm",
        battery_life = 3.0, 
        charging_time = 3.0,
        connectivity = "Bluetooth",
        wireless_range= 10.0,
        sensor_type = "Dry",
        sensor_material = None,
        extra_sensors = "9 axis IMU",
        raw_data_access= "Real-time, On-board storage (8GB)",
        raw_data_availability= "SDK",
        price = 14200.0,
        product_link = "https://www.bitbrain.com/neurotechnology-products/dry-eeg/hero"
)
BITBRAIN_IKON = HeadsetProfile(
        name = "BITBRAIN_IKON",
        channels = [
                 'Fp1', 'Fpz', 'Fp2',
            'AF7',                  'AF8'
        ],
        label = "Bitbrain Ikon 5 ch",
        company_name="Bitbrain",
        form_factor= "Headband",
        sampling_rate = [256.0],
        bit_depth = 24,
        reference = ['A1', 'A2'],
        montage = "10-20",
        bandwidth = (0.0, 40.0),
        weight= 100.0,
        size = "Adjustable 52-72 cm",
        battery_life = 9.0,
        charging_time= 3.0,
        connectivity = "Bluetooth",
        wireless_range= None,
        sensor_type = "Dry",
        sensor_material = None,
        extra_sensors = "9 axis IMU",
        raw_data_access= "Real-time, On-board storage (8GB)",
        raw_data_availability= "SDK",
        price = 2500.0,
        product_link = "https://www.bitbrain.com/neurotechnology-products/textile-eeg/ikon"
)

#----------- MUSE HEADSETS -------------#
MUSE_2_HEADBAND = HeadsetProfile(
        name = "MUSE_2_HEADBAND",
        channels = [
             'AF7',      'AF8',
             'TP9',      'TP10'
        ],
        company_name = "InteraXon",
        label = "Muse 2 headband 4 ch",
        form_factor = "Helmet",
        sampling_rate = [256.0],
        bit_depth = 12,
        reference = ['Fpz'],
        montage = "10-20",
        bandwidth = None,
        weight = 38.5,
        size = "Adjustable 30-35cm, ear to ear",
        battery_life = 5.0,
        charging_time = 3.0,
        connectivity = "Bluetooth",
        wireless_range = 10.0,
        sensor_type = "Dry",
        sensor_material = "Conductive gold",
        extra_sensors = "Heart (PPG), IMU with accelerometer and gyroscope",
        raw_data_access= "Real-time",
        raw_data_availability= "Paid App, SDK",
        price = 270.0,
        product_link = "https://eu.choosemuse.com/"
)

MUSE_S_ATHENA = HeadsetProfile(
        name = "MUSE_S_ATHENA",
        channels = [
             'AF7',      'AF8',
             'TP9',      'TP10'
        ],
        company_name = "InteraXon",
        label = "Muse S Athena 4 ch",
        form_factor = "Headband",
        sampling_rate = [256.0],
        bit_depth = 14,
        reference = ['Fpz'],
        montage = "10-20",
        bandwidth = None,
        weight = 41.0,
        size = "Flexible, 43-63 cm",
        battery_life = 3.0,
        charging_time = 3.0,
        connectivity = "Bluetooth",
        wireless_range = 10.0,
        sensor_type = "Dry",
        sensor_material = "Conductive silver thread",
        extra_sensors = "Heart (PPG), brain oxygenation (fNIRS), IMU with accelerometer and gyroscope",
        raw_data_access= "Real-time",
        raw_data_availability= "Paid App, SDK",
        price = 450.0,
        product_link = "https://eu.choosemuse.com/"
)

#----------- NEUROSITY HEADSETS -----------#
NEUROSITY_CROWN = HeadsetProfile (
         name = "NEUROSITY_CROWN",
         channels = [ 
              'F5',    'F6',
                'C3', 'C4',
               'CP3', 'CP4',
               'PO3', 'PO4'
         ],
         label = "Neurosity crown 8 ch",
         company_name = "Neurosity",
         form_factor = "Helmet",
         sampling_rate = [256.0],
         bit_depth = None,
         reference = ['T7', 'T8'],
         montage = "10-20",
         bandwidth = None,
         weight = 250.0,
         size = "16.5 x 15 x 8 cm",
         battery_life = 3.0, 
         charging_time = None,
         connectivity= "Bluetooth, BLE, Cloud, USBC, WiFi",
         wireless_range = None, 
         sensor_type = "Dry",
         sensor_material = "Ag/AgCl",
         extra_sensors = "On-board processor, accelerometer, 2 haptic motors",
         raw_data_access= "Real-time, on-board storage (8GB)",
         raw_data_availability= "Paid App, SDK",
         price = 1500.0,
         product_link = "https://neurosity.co/crown"
)

#----------- CGX HEADSETS -----------#
CGX_QUICK_20R = HeadsetProfile (
         name = "CGX_QUICK_20R",
         channels = [
                     'Fp1', 'Fp2',
              'F7', 'F3', 'Fz', 'F4', 'F8',
              'T7', 'C3', 'Cz', 'C4', 'T8',
              'P7', 'P3', 'Pz', 'P4', 'P8',
                       'O1', 'O2'
         ],
         label = "CGX Quick 20r o m",
         company_name= "CGX Systems",
         form_factor= "Helmet",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['A1', 'A2'],
         montage = "10-20",
         bandwidth = (0.0, 131.0),
         weight = 526.0,
         size = "20 x 18 x 19 cm",
         battery_life = 8.0,
         charging_time = None,
         connectivity= "Bluetooth LE",
         wireless_range = None,
         sensor_type = "Dry",
         sensor_material = "Ag/AgCl",
         extra_sensors = "Accelerometer, 2 extra ExG electrodes to assign as desired",
         raw_data_access= "Real-time",
         raw_data_availability= "Streaming API",
         price = 30000.0,
         product_link = "https://www.cgxsystems.com/quick-20r-v2"
)
CGX_QUICK_32R = HeadsetProfile (
         name = "CGX_QUICK_32R",
         channels = [
                  'Fp1', 'Fpz', 'Fp2',
              'AF7',                 'AF8',
              'F7', 'F3', 'Fz', 'F4', 'F8',
                 'FC5',            'FC6',
              'T7', 'C3', 'Cz', 'C4', 'T8',
                  'CP5',          'CP6',
              'P7', 'P3', 'Pz', 'P4', 'P8',
               'PO7',              'PO8',
                    'O1', 'Oz', 'O2'
         ],
         label = "CGX Quick 32r o m",
         company_name= "CGX Systems",
         form_factor= "Helmet",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['A1', 'A2'],
         montage = "10-20",
         bandwidth = (0.0, 131.0),
         weight = 646.0,
         size = "20 x 18 x 19 cm",
         battery_life = 8.0,
         charging_time = None,
         connectivity= "Bluetooth LE",
         wireless_range = None,
         sensor_type = "Dry",
         sensor_material = "Ag/AgCl",
         extra_sensors = "Accelerometer, 2 extra ExG electrodes to assign as desired",
         raw_data_access= "Real-time",
         raw_data_availability= "Streaming API",
         price = 35000.0,
         product_link = "https://www.cgxsystems.com/quick-32r"
)

#----------- MINDROVE -----------#
MINDROVE_VISION = HeadsetProfile (
         name = "MINDROVE_VISION",
         channels = [
                     'Fp1', 'Fp2',
                'C5', 'C1',  'C2', 'C6',
                      'O1',  'O2'
         ],
         company_name = "MindRove",
         label = "MindRove vision 8 channels",
         form_factor = "Headband",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['TP9', 'TP10'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "WiFi direct",
         wireless_range= None,
         battery_life= 6.0,
         charging_time= None,
         sensor_type = "Dry",
         sensor_material = "Dry polymer electrodes with Ag/AgCl coating",
         extra_sensors = "6 axis IMU",
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         price = 1000.0,
         product_link = "https://mindrove.com/product/vision-8-channel-eeg/"
)
MINDROVE_LUCID = HeadsetProfile (
         name = "MINDROVE_LUCID",
         label = "MindRove Lucid 6 channels",
         channels = [
                     'Fp1', 'Fp2',
                      'C1',  'C2', 
                      'O1',  'O2'
         ],
         company_name= "MindRove",
         form_factor = "Headband",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['TP9', 'TP10'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "WiFi direct",
         wireless_range= None,
         battery_life= 6.0,
         charging_time= None,
         sensor_type = "Dry",
         sensor_material = "Dry polymer electrodes with Ag/AgCl coating",
         extra_sensors = "6 axis IMU",
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         price = 762.0,
         product_link = "https://mindrove.com/product/lucid-6-channel-eeg-headset/"
)
MINDROVE_ARC = HeadsetProfile (
         name = "MINDROVE_ARC",
         label = "MindRove Arc 6 channels",
         channels = [
                   'C5', 'C3', 'C1', 'C2', 'C4', 'C6'
         ],
         company_name= "MindRove",
         form_factor = "Helmet",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['TP9', 'TP10'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "WiFi direct",
         wireless_range= None,
         battery_life= 5.0,
         charging_time= None,
         sensor_type = "Wet saline",
         sensor_material = "Conductive fabric with platinum threads",
         extra_sensors = "6 axis IMU",
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         price = 745.0,
         product_link = "https://mindrove.com/arc/"
)
MINDROVE_BRIGHT = HeadsetProfile (
         name = "MINDROVE_BRIGHT",
         label = "MindRove Bright 4 channels, headband",
         channels = [
                     'Fp1', 'Fp2',
                      'O1',  'O2'         
        ],
         company_name= "MindRove",
         form_factor = "Headband",
         sampling_rate = [500.0],
         bit_depth = 24,
         reference = ['TP9', 'TP10'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "WiFi direct",
         wireless_range= None,
         battery_life= 6.0,
         charging_time= None,
         sensor_type = "Dry",
         sensor_material = "Dry polymer electrodes with Ag/AgCl coating",
         extra_sensors = "6 axis IMU",
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         price = 615.0,
         product_link = "https://mindrove.com/product/bright-4-channel-eeg-headset/"
)

#----------- BRAINBIT -----------#
BRAINBIT_DRAGON = HeadsetProfile (
         name = "BRAINBIT_DRAGON",
         label = "Brainbit DragonEEG 21 ch",
         channels = [
                       'Fp1', 'Fpz', 'Fp2', 
             'F7',       'F3', 'Fz', 'F4',       'F8', 
             'P7', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P8',
                         'P3', 'Pz', 'P4', 
                         'O1', 'Oz', 'O2'
         ],
         company_name= "Brainbit",
         form_factor = "Cap",
         sampling_rate = [500.0],
         bit_depth = None,
         reference = None,
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight = None,
         size = "3 caps (S, M, L)",
         connectivity= "Bluetooth LE",
         wireless_range= None,
         battery_life=None,
         charging_time=None,
         sensor_type = "Dry or Wet",
         sensor_material = "Dry golden plated spring-loaded electrodes OR Ag/AgCl electrodes",
         extra_sensors = "3 poly channels for EMG, ECG, EOG",
         raw_data_access = "Real-time",
         raw_data_availability = "Paid app, SDK",
         price = 3218.0,
         product_link = "https://store.brainbit.com/collections/hardware/products/dragoneeg"
)
BRAINBIT_HEADPHONES = HeadsetProfile (
         name = "BRAINBIT_HEADPHONES",
         label = "Brainbit Headphones 3 ch",
         channels = [
                       'C3', 'Cz', 'C4'
         ],
         company_name = "Brainbit",
         form_factor= "Headphones",
         sampling_rate = [250.0],
         bit_depth = None,
         reference = ['A1', 'A2'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "Bluetooth",
         wireless_range= None,
         battery_life= None,
         charging_time=None,
         sensor_type = "Dry",
         sensor_material = "Dry golden plated spring-loaded electrodes",
         extra_sensors = "Microphone",
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         notes = "Two versions, Pro and Lite",
         price = 790.0,
         product_link = "https://store.brainbit.com/collections/hardware/products/brainbit-headphones"
)
BRAINBIT_HEADBAND_PRO = HeadsetProfile (
         name = "BRAINBIT_HEADBAND_PRO",
         label = "Headband Pro 8 channels",
         channels = [
                          'Fp1', 'Fp2', 
                        'C3',       'C4',
                        'P7',       'P8',
                            'O1', 'O2'
         ],
         company_name= "Brainbit",
         form_factor= "Headband",
         sampling_rate = [250.0],
         bit_depth = None,
         reference = ['A1', 'A2'],
         montage = "10-20",
         bandwidth = (0.0, 250.0),
         weight= None,
         size= None,
         connectivity= "Bluetooth",
         wireless_range= None,
         battery_life= None,
         charging_time=None,
         sensor_type = "Dry",
         sensor_material = "Snap-in on gold-plated pogo pins",
         extra_sensors = "Swap C3 and C4 for P7 and P8",
         raw_data_access = "Real-time",
         raw_data_availability = "App, SDK",
         price = 1042.0,
         product_link = "https://store.brainbit.com/collections/hardware/products/headband-pro"
)
BRAINBIT_HEADBAND = HeadsetProfile (
         name = "BRAINBIT_HEADBAND",
         label = "Headband 4 channels",
         channels = [
                     'T7',  'T8',
                      'O1', 'O2'
         ],
         company_name= "Brainbit",
         form_factor= "Headband",
         sampling_rate = [250.0],
         bit_depth = None,
         reference = ['A1', 'A2'],
         montage = "10-20",
         bandwidth = (0.0, 100.0),
         weight= None,
         size= None,
         battery_life= 12.0,
         charging_time= 4.0,
         connectivity= "Bluetooth LE",
         sensor_type = "Dry",
         sensor_material = "Dry gold-plated spring-loaded electrodes",
         extra_sensors = None,
         raw_data_access = "Real-time",
         raw_data_availability = "SDK",
         price = 434.0,
         product_link = "https://store.brainbit.com/collections/hardware/products/brainbit-sdk"
)

# ----------------NEEURO HEADSETS---------------- #
NEEURO_SENZEBAND = HeadsetProfile(
    name = "NEEURO_SENZEBAND",
    channels = [
                    'Fp1', 'Fp2',
                'T3',         'T4'
    ],
    company_name = "Neeuro",
    label = "Neeuro Senzeband 2 4 ch",
    form_factor = "Headband",
    sampling_rate = [250.0],
    bit_depth = 24,
    reference = ['Fpz'], 
    montage = "10-20",
    bandwidth = None,
    weight= 88.0,
    size = "17.9cm (L) x 15.1cm (W) x 3.1cm (H)",
    battery_life = 6.0,
    charging_time = 2.0,
    connectivity = "Bluetooth 5.0",
    wireless_range = None,
    sensor_type = "Dry",
    sensor_material = "Proprietary elastomer sensors",
    extra_sensors = "9 axis IMU, PPG sensor, SpO2 sensor",
    raw_data_access = "Real-time",
    raw_data_availability = "Support 3rd party apps",
    price = 450.0,
    product_link = "https://www.neeuro.com/senzeband/product-features"
)

# ----------------ADVANCED BRAIN MONITORING HEADSETS---------------- #
ABM_B_ALERT_X10 = HeadsetProfile(
    name = "ABM_B_ALERT_X10",
    channels = [
                'F3',  'Fz',  'F4',
                'C3',  'Cz',  'C4',
                'P3',  'POz',  'P4'
    ],
    company_name = "ABM",
    label = "Advanced Brain Monitoring B-Alert X10 9 ch",
    form_factor = "Helmet",
    sampling_rate = [256.0],
    bit_depth = 16,
    reference = ['LM'], #linked mastoids
    montage = "10-20",
    bandwidth = (0.1, 67.0),
    weight= 110.0,
    size = "6,86 cm (L) x 4,83 cm (W) x 2,03 cm (H)",
    battery_life = 9.0,
    charging_time = None,
    connectivity = "Bluetooth 2.1",
    wireless_range = 10.0,
    sensor_type = "Wet",
    sensor_material = "Foam Sensor (100 PPI Natural Color Filter Foam)",
    extra_sensors = "2 optional EOG, ECG, EMG channels",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid app",
    price = None,
    product_link = "https://www.advancedbrainmonitoring.com/products/b-alert-mobile"
)
ABM_B_ALERT_X24 = HeadsetProfile(
    name = "ABM_B_ALERT_X24",
    channels = [
                   'Fp1',  'Fp2',
           'F7', 'F3',  'Fz',  'F4', 'F8',
           'T7', 'C3',  'Cz',  'C4', 'T8', #old T3, T4
           'P7', 'P3',  'Pz',  'P4', 'P8',
                  'O1','POz','O2'
    ],
    company_name = "ABM",
    label = "Advanced Brain Monitoring B-Alert X24 20 ch",
    form_factor = "Helmet",
    sampling_rate = [256.0],
    bit_depth = 16,
    reference = ['LM'], #linked mastoids
    montage = "10-20",
    bandwidth = (0.1, 67.0),
    weight= 56.7,
    size = "6,86 cm (L) x 4,83 cm (W) x 2,03 cm (H)",
    battery_life = 8.0,
    charging_time = None,
    connectivity = "Bluetooth 2.1",
    wireless_range = 10.0,
    sensor_type = "Wet",
    sensor_material = "Foam Sensor (100 PPI Natural Color Filter Foam)",
    extra_sensors = "2 optional EOG, ECG, EMG channels",
    raw_data_access = "Real-time",
    raw_data_availability = "Paid app",
    price = None,
    product_link = "https://www.advancedbrainmonitoring.com/products/b-alert-mobile"
)

# ----------------NEUROSKY HEADSETS---------------- #
NEUROSKY_MINDWAVE_2 = HeadsetProfile(
    name = "NEUROSKY_MINDWAVE_2",
    channels = [
                'Fp1'
    ],
    company_name = "NeuroSky",
    label = "NeuroSky MindWave Mobile 2 EEG Sensor",
    form_factor = "Helmet",
    sampling_rate = [512.0],
    bit_depth = 12,
    reference = ['A1'], #left earlobe
    montage = "10-20",
    bandwidth = (3, 100.0),
    weight= 90.0,
    size = "22.5 (H) x 15.5 (W) x 9.2 (D) cm",
    battery_life = 8.0,
    charging_time = None,
    connectivity = "Bluetooth, BLE",
    wireless_range = 10.0,
    sensor_type = "Dry",
    sensor_material = None,
    extra_sensors = None,
    raw_data_access = "Real-time",
    raw_data_availability = "App, SDK",
    price = 150.0,
    product_link = "https://store.neurosky.com/pages/mindwave"
)

# ----------------WEARABLE SENSING HEADSETS---------------- #
WS_DSI_24 = HeadsetProfile(
    name = "WS_DSI_24",
    channels = [
                    'Fp1', 'Fp2',
          'F7',  'F3',  'Fz',  'F4',  'F8',
          'T7',  'C3',  'Cz',  'C4',  'T8',
          'P7',  'P3',  'Pz',  'P4',  'P8',
                    'O1',   'O2'
    ],
    company_name = "Wearable Sensing",
    label = "Wearable Sensing DSI-24 19 ch",
    form_factor = "Helmet",
    sampling_rate = [300.0],
    bit_depth = 16,
    reference = ['Fpz', 'A1', 'A2'], #left and right earlobes
    montage = "10-20",
    bandwidth = (0.003, 150.0),
    weight= None,
    size = "Adult version: 52-62 cm circumference",
    battery_life = 24.0,
    charging_time = None,
    connectivity = "Bluetooth",
    wireless_range = 10.0,
    sensor_type = "Dry",
    sensor_material = None,
    extra_sensors = "3 additional inputs for EMG, ECG, EOG, 3 axis accelerometer",
    raw_data_access = "Real-time",
    raw_data_availability = "App, API",
    price = None,
    product_link = "https://wearablesensing.com/dsi-24/"
)
WS_DSI_7 = HeadsetProfile(
    name = "WS_DSI_7",
    channels = [
                 'F3',     'F4',
                 'C3',     'C4',  
                 'P3','Pz','P4', 
    ],
    company_name = "Wearable Sensing",
    label = "Wearable Sensing DSI-7 7 ch",
    form_factor = "Helmet",
    sampling_rate = [300.0],
    bit_depth = 16,
    reference = ['A1', 'A2'], #left and right earlobes
    montage = "10-20",
    bandwidth = (0.003, 150.0),
    weight= None,
    size = "Adult version: 52-62 cm circumference",
    battery_life = 24.0,
    charging_time = None,
    connectivity = "Bluetooth",
    wireless_range = 10.0,
    sensor_type = "Dry",
    sensor_material = None,
    extra_sensors = "3 additional inputs for EMG, ECG, EOG, 3 axis accelerometer",
    raw_data_access = "Real-time",
    raw_data_availability = "App, API",
    price = None,
    notes="Customizable positions for the 7 electrodes",
    product_link = "https://wearablesensing.com/dsi-7/"
)

WS_DSI_VR300 = HeadsetProfile(
    name = "WS_DSI_VR300",
    channels = [
                     'FCz', 
                 'P3','Pz','P4', 
            'PO7',             'PO8',
                      'Oz'
    ],
    company_name = "Wearable Sensing",
    label = "Wearable Sensing DSI-VR300 7 ch",
    form_factor = "Helmet",
    sampling_rate = [300.0],
    bit_depth = 16,
    reference = ['A1', 'A2'], #left and right earlobes
    montage = "10-20",
    bandwidth = (0.003, 150.0),
    weight= None,
    size = "Adult version: 52-62 cm circumference",
    battery_life = 12.0,
    charging_time = None,
    connectivity = "Bluetooth",
    wireless_range = 10.0,
    sensor_type = "Dry",
    sensor_material = None,
    extra_sensors = None,
    raw_data_access = "Real-time",
    raw_data_availability = "App, API",
    price = None,
    notes="Designed for VR applications, visual stimuli and P300 experiments",
    product_link = "https://wearablesensing.com/dsi-vr300/"
)
WS_DSI_VRVEP = HeadsetProfile(
    name = "WS_DSI_VRVEP",
    channels = [
                       'FCz', 
                 'PO3','POz','PO4', 
                  'O1', 'Oz', 'O2'
    ],
    company_name = "Wearable Sensing",
    label = "Wearable Sensing DSI-VRVEP 7 ch",
    form_factor = "Helmet",
    sampling_rate = [300.0],
    bit_depth = 16,
    reference = ['A1', 'A2'], #left and right earlobes
    montage = "10-20",
    bandwidth = (0.003, 150.0),
    weight= None,
    size = "Adult version: 52-62 cm circumference",
    battery_life = 12.0,
    charging_time = None,
    connectivity = "Bluetooth",
    wireless_range = 10.0,
    sensor_type = "Dry",
    sensor_material = None,
    extra_sensors = None,
    raw_data_access = "Real-time",
    raw_data_availability = "App, API",
    price = None,
    notes="Designed for VR applications, visual processing and SSVEP experiments",
    product_link = "https://wearablesensing.com/dsi-vr300/"
)

# Dictionary of available headsets
HEADSETS: Dict[str, HeadsetProfile] = {
    "FULL_CHO": FULL_CHO,
    "EMOTIV_FLEX": EMOTIV_FLEX,
    "EMOTIV_EPOCH_X": EMOTIV_EPOCH_X,
    "EMOTIV_INSIGHT": EMOTIV_INSIGHT,
    "EMOTIV_MN8": EMOTIV_MN8,
    "ULTRACORTEX_16": ULTRACORTEX_16,
    "ULTRACORTEX_8": ULTRACORTEX_8,
    "GTEC_UNICORN_HYBRID": GTEC_UNICORN_HYBRID,
    "BITBRAIN_DIADEM": BITBRAIN_DIADEM,
    "BITBRAIN_AIR": BITBRAIN_AIR,
    "BITBRAIN_HERO": BITBRAIN_HERO,
    "BITBRAIN_IKON": BITBRAIN_IKON,
    "MUSE_2_HEADBAND": MUSE_2_HEADBAND,
    "MUSE_S_ATHENA": MUSE_S_ATHENA,
    "NEUROSITY_CROWN": NEUROSITY_CROWN,
    "CGX_QUICK_20R": CGX_QUICK_20R,
    "CGX_QUICK_32R": CGX_QUICK_32R,
    "MINDROVE_VISION": MINDROVE_VISION,
    "MINDROVE_LUCID": MINDROVE_LUCID,
    "MINDROVE_ARC": MINDROVE_ARC,
    "MINDROVE_BRIGHT": MINDROVE_BRIGHT,
    "BRAINBIT_DRAGON": BRAINBIT_DRAGON,
    "BRAINBIT_HEADPHONES": BRAINBIT_HEADPHONES,
    "BRAINBIT_HEADBAND_PRO": BRAINBIT_HEADBAND_PRO,
    "BRAINBIT_HEADBAND":BRAINBIT_HEADBAND,
    "NEEURO_SENZEBAND": NEEURO_SENZEBAND,
    "ABM_B_ALERT_X10": ABM_B_ALERT_X10,
    "ABM_B_ALERT_X24": ABM_B_ALERT_X24,
    "NEUROSKY_MINDWAVE_2": NEUROSKY_MINDWAVE_2,
    "WS_DSI_24": WS_DSI_24,
    "WS_DSI_7": WS_DSI_7,
    "WS_DSI_VR300": WS_DSI_VR300,
    "WS_DSI_VRVEP": WS_DSI_VRVEP
}

# --------------------------- Utility functions --------------------------- #

# Print a compact catalogue of available headset profiles.
def print_catalog() -> None:
    # --- column widths ---
    COL_COMPANY = 14
    COL_PRODUCT = 26
    COL_FORM = 12
    COL_NCH = 4
    COL_CHANNELS = 54
    COL_PRICE = 12   # <<< compactness

    TOTAL_WIDTH = (
        COL_COMPANY + 2 +
        COL_PRODUCT + 2 +
        COL_FORM + 2 +
        COL_NCH + 2 +
        COL_CHANNELS + 1 +
        COL_PRICE
    )

    print("=" * TOTAL_WIDTH)
    print(
        f"{'Company':<{COL_COMPANY}}  "
        f"{'Product':<{COL_PRODUCT}}  "
        f"{'Form':<{COL_FORM}}  "
        f"{'n_ch':>{COL_NCH}}  "
        f"{'Channels':<{COL_CHANNELS}} "
        f"{'Price':>{COL_PRICE}}"
    )
    print("-" * TOTAL_WIDTH)

    items_sorted = sorted(
        HEADSETS.items(),
        key=lambda kv: (
            (kv[1].company_name or "").lower(),
            -(kv[1].nr_channels if hasattr(kv[1], "nr_channels") else len(getattr(kv[1], "channels", []))),
            (kv[1].name or "").lower(),
        ),
    )

    for _, profile in items_sorted:
        company = (profile.company_name or "-")
        product = (profile.name or "-")
        form = (getattr(profile, "form_factor", None) or "-")

        chs = list(getattr(profile, "channels", []))
        n_ch = len(chs)
        ch_str = ", ".join(chs)

        wrap_width = COL_CHANNELS
        price_str = str(profile.price) if getattr(profile, "price", None) is not None else "-"

        lines = []
        while len(ch_str) > wrap_width:
            cut = ch_str.rfind(",", 0, wrap_width)
            if cut == -1:
                cut = wrap_width
            lines.append(ch_str[:cut].strip())
            ch_str = ch_str[cut + 1 :].strip()
        lines.append(ch_str)

        # first row with company, product, form, n_ch, first line of channels, price
        print(
            f"{company:<{COL_COMPANY}}  "
            f"{product:<{COL_PRODUCT}}  "
            f"{form:<{COL_FORM}}  "
            f"{n_ch:>{COL_NCH}}  "
            f"{lines[0]:<{COL_CHANNELS}} "
            f"{price_str:>{COL_PRICE}}"
        )

        # continuation rows for wrapped channel lists
        for extra in lines[1:]:
            print(
                f"{'':<{COL_COMPANY}}  "
                f"{'':<{COL_PRODUCT}}  "
                f"{'':<{COL_FORM}}  "
                f"{'':>{COL_NCH}}  "
                f"{extra:<{COL_CHANNELS}} "
                f"{'':>{COL_PRICE}}"
            )

        print("-" * TOTAL_WIDTH)

    print("=" * TOTAL_WIDTH)
    print(f"Total headsets: {len(HEADSETS)}")

# Remove occipital / parieto-occipital channels from available channels
def make_no_occipital_mask(available_channels: List[str]) -> List[str]:
    # remove channels whose names start with O or PO
    return [
        ch for ch in available_channels
        if not (ch.startswith('O') or ch.startswith('PO'))
    ]

# print_catalog()

# Backward-compatible alias used by earlier scripts.
print_headsets_list = print_catalog
