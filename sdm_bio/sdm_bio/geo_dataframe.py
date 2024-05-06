
"""_summary_

mdata <- sdmData(
  formula = harpia ~ .,
  train = sp_thin,
  predictors = bio_harpia,
  bg = list(
    n = 10000,
    method = "gRandom",
    remove = TRUE
  )
)

bio <- crop(bio, br) # recorte da área de estudo
bio <- mask(bio, br) # máscara fora da área de estudo
"""
class MapLayer:
    def __init__(self):
        pass
    
class Bounds:
    def __ini__(self, shp: str):
        pass