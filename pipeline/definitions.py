from . import assets 
from dagster import Definitions, load_assets_from_modules 

defs = Definitions(
    assets=load_assets_from_modules([assets])
)