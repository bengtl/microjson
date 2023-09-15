"""Microjson GeoJSON auto-generated by datamodel-codegen"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Type(BaseModel):
    type: str
    enum: List[str]


class Items3(BaseModel):
    type: str


class Items2(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items3] = None


class Items1(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items2] = None


class Items(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items1] = None


class Coordinates(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items


class Items4(Items3):
    pass


class Bbox(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items4


class Type1(Type):
    pass


class Items9(Items3):
    pass


class Items8(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items9] = None


class Items7(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items8] = None


class Items6(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items7] = None


class Coordinates1(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items6


class Items10(Items3):
    pass


class Bbox1(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items10


class Properties1(BaseModel):
    type: Type1
    coordinates: Coordinates1
    bbox: Bbox1


class OneOfItem1(BaseModel):
    title: str
    type: str
    required: List[str]
    properties: Properties1


class Items5(BaseModel):
    one_of: List[OneOfItem1] = Field(..., alias='oneOf')


class Geometries(BaseModel):
    type: str
    items: Items5


class OneOfItem2(Items3):
    pass


class Id(BaseModel):
    one_of: List[OneOfItem2] = Field(..., alias='oneOf')


class OneOfItem3(Items3):
    pass


class Properties2(BaseModel):
    one_of: List[OneOfItem3] = Field(..., alias='oneOf')


class Type2(Type):
    pass


class Items14(Items3):
    pass


class Items13(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items14] = None


class Items12(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items13] = None


class Items11(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items12] = None


class Coordinates2(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items11


class Items15(Items3):
    pass


class Bbox2(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items15


class Type3(Type):
    pass


class Items20(Items3):
    pass


class Items19(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items20] = None


class Items18(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items19] = None


class Items17(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items18] = None


class Coordinates3(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items17


class Items21(Items3):
    pass


class Bbox3(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items21


class Properties4(BaseModel):
    type: Type3
    coordinates: Coordinates3
    bbox: Bbox3


class OneOfItem5(BaseModel):
    title: str
    type: str
    required: List[str]
    properties: Properties4


class Items16(BaseModel):
    one_of: List[OneOfItem5] = Field(..., alias='oneOf')


class Geometries1(BaseModel):
    type: str
    items: Items16


class Properties3(BaseModel):
    type: Type2
    coordinates: Optional[Coordinates2] = None
    bbox: Bbox2
    geometries: Optional[Geometries1] = None


class OneOfItem4(BaseModel):
    type: str
    title: Optional[str] = None
    required: Optional[List[str]] = None
    properties: Optional[Properties3] = None


class Geometry(BaseModel):
    one_of: List[OneOfItem4] = Field(..., alias='oneOf')


class Type4(Type):
    pass


class OneOfItem6(Items3):
    pass


class Id1(BaseModel):
    one_of: List[OneOfItem6] = Field(..., alias='oneOf')


class OneOfItem7(Items3):
    pass


class Properties6(BaseModel):
    one_of: List[OneOfItem7] = Field(..., alias='oneOf')


class Type5(Type):
    pass


class Items26(Items3):
    pass


class Items25(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items26] = None


class Items24(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items25] = None


class Items23(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items24] = None


class Coordinates4(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items23


class Items27(Items3):
    pass


class Bbox4(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items27


class Type6(Type):
    pass


class Items32(Items3):
    pass


class Items31(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items32] = None


class Items30(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items31] = None


class Items29(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Optional[Items30] = None


class Coordinates5(BaseModel):
    type: str
    min_items: Optional[int] = Field(None, alias='minItems')
    items: Items29


class Items33(Items3):
    pass


class Bbox5(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items33


class Properties8(BaseModel):
    type: Type6
    coordinates: Coordinates5
    bbox: Bbox5


class OneOfItem9(BaseModel):
    title: str
    type: str
    required: List[str]
    properties: Properties8


class Items28(BaseModel):
    one_of: List[OneOfItem9] = Field(..., alias='oneOf')


class Geometries2(BaseModel):
    type: str
    items: Items28


class Properties7(BaseModel):
    type: Type5
    coordinates: Optional[Coordinates4] = None
    bbox: Bbox4
    geometries: Optional[Geometries2] = None


class OneOfItem8(BaseModel):
    type: str
    title: Optional[str] = None
    required: Optional[List[str]] = None
    properties: Optional[Properties7] = None


class Geometry1(BaseModel):
    one_of: List[OneOfItem8] = Field(..., alias='oneOf')


class Items34(Items3):
    pass


class Bbox6(BaseModel):
    type: str
    min_items: int = Field(..., alias='minItems')
    items: Items34


class Properties5(BaseModel):
    type: Type4
    id: Id1
    properties: Properties6
    geometry: Geometry1
    bbox: Bbox6


class Items22(BaseModel):
    title: str
    type: str
    required: List[str]
    properties: Properties5


class Features(BaseModel):
    type: str
    items: Items22


class Properties(BaseModel):
    type: Type
    coordinates: Optional[Coordinates] = None
    bbox: Bbox
    geometries: Optional[Geometries] = None
    id: Optional[Id] = None
    properties: Optional[Properties2] = None
    geometry: Optional[Geometry] = None
    features: Optional[Features] = None


class OneOfItem(BaseModel):
    title: str
    type: str
    required: List[str]
    properties: Properties


class Model(BaseModel):
    _schema: str = Field(..., alias='$schema')
    _id: str = Field(..., alias='$id')
    title: str
    one_of: List[OneOfItem] = Field(..., alias='oneOf')
