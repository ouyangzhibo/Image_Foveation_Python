# Image Foveation Python
Python implementation of image retina transformation. For detailed algorithm, please refer to this paper:

> Perry, Jeffrey S., and Wilson S. Geisler. "Gaze-contingent real-time simulation of arbitrary visual fields." Human vision and electronic imaging VII. Vol. 4662. International Society for Optics and Photonics, 2002.


Prerequisites
---
- Python 3.x
- Numpy
- OpenCV

Usage
---
```
python retina_transform.py [image_path]
```

To adjust the degree of blur and size of the foveal region (full-reolustion pixels), one can increase or decrease the value of ```k``` and ```p``` and ```alpha``` in the function ```foveat_img```. 

Example
---
<img src="images/000000000139.jpg" width="420"> <img src="images/000000000139_RT.jpg" width="420">
