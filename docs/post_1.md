🚀 **Decoding the Atmosphere: Processing 6-Channel Raw INSAT-3D Satellite Data from Scratch!** 🌍🛰️

Over the last few days, I've been diving deep into raw Level-1B (L1B) HDF5 data from the INSAT-3D meteorological satellite, building a custom pipeline to process and visualize its telemetry using purely Python (`h5py` and `numpy`). 

What started as parsing a single Thermal Infrared channel turned into a full-scale pipeline to decode and align all 6 available imager channels simultaneously into a synchronized time-lapse video grid! 

Here are a few technical challenges and wins from the project:

📊 **Translating Telemetry to Physics**
Raw satellite data comes down as 10-bit digital numbers. To make it meaningful, the script decodes these values using the HDF5 lookup tables to map them directly to physical variables—converting Visible and Shortwave Infrared (SWIR) into **Albedo percentages**, and mapping Thermal and Water Vapor channels into **Kelvin Brightness Temperatures**.

🛰️ **INSAT-3D and Its 6 Imager Channels**
INSAT-3D is an advanced Indian meteorological satellite parked in geostationary orbit 36,000 km above the equator. Its payload includes a versatile 6-channel imager that constantly scans the Earth. My pipeline processes all six of them simultaneously into a synchronized 2x3 grid:
- **VIS (Visible):** Tracks physical cloud cover and weather systems during the day.
- **SWIR (Shortwave IR):** Essential for distinguishing between water droplets (bright) and ice crystals (dark).
- **MIR (Mid IR):** Detects fog, low clouds, and even active thermal anomalies like forest fires.
- **WV (Water Vapor):** Maps atmospheric moisture patterns and jet streams in the mid-to-upper troposphere.
- **TIR-1 & TIR-2 (Thermal IR):** Measures exact cloud-top and sea surface temperatures, working 24/7 both day and night!

There is something incredibly satisfying about pulling raw, compressed arrays of numbers from a spacecraft 36,000 km away and turning them into a living visualization of the planet.

Next up: diving deeper into specific atmospheric phenomena hidden inside this data!

#Meteorology #SatelliteImagery #Python #DataScience #EarthObservation #INSAT #RemoteSensing #HDF5
