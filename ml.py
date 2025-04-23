import json
from datetime import date
from pydantic import BaseModel
from pydantic import ValidationError
import numpy as np
import pandas as pd

class PromoData(BaseModel):
    promo_start: date
    promo_end: date
    shipping_start: date
    shipping_end: date
    promo_type: str
    feat_2: float
    feat_3: float
    agent: str
    promo_id: str
    item_id: str
    promo_class: str
    feat_7: float
    feat_9: float
    feat_10: float
    feat_11: float
    feat_12: int


def load_json_to_model(file_path: str) -> PromoData:
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      json_data = json.load(f)

    promo_data = PromoData(**json_data)
    return promo_data

  except json.JSONDecodeError as e:
    print(f"Ошибка парсинга JSON: {e}")
  except ValidationError as e:
    print(f"Ошибка валидации данных: {e.errors()}")
  except FileNotFoundError:
    print(f"Файл {file_path} не найден")
  return None

def preproc(input : PromoData) -> pd.Series:

  cat = ['promo_type', 'promo_class', 'agent', 'promo_id', 'item_id']
  num = ['feat_2', 'feat_3','feat_7', 'feat_9', 'feat_10', 'feat_11', 'feat_12', 'day_sin', 'day_cos', 'left_offset',
               'right_offset', 'promo_duration', 'shipping_duration']
  D = dict(input)
  D['promo_duration'] = (input.promo_end - input.promo_start).days
  D['shipping_duration'] = (input.shipping_end - input.shipping_start).days
  D['right_offset'] = (input.promo_end - input.promo_start).days
  D['left_offset'] = (input.shipping_end - input.shipping_start).days
  D['promo_heart'] = (D['promo_start'] + (D['promo_end'] - D['promo_start']) // 2)
  D['promo_heart'] = D['promo_heart'].day + (D['promo_heart'].month-1)*30
  D['day_sin'] = np.sin(2 * np.pi * D['promo_heart']/365)
  D['day_cos'] = np.cos(2 * np.pi * D['promo_heart']/365)
  return pd.Series(D)[cat+num]