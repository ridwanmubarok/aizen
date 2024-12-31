from datetime import datetime

def get_season(location, date):
    # Indonesia
    if 'Indonesia' in location:
        if date.month >= 4 and date.month <= 10:
            return 'indonesia_dry'
        return 'indonesia_wet'
    
    # Japan
    elif 'Japan' in location or any('\u4e00' <= char <= '\u9fff' for char in location):
        if 3 <= date.month <= 5:
            return 'japan_spring'
        elif 6 <= date.month <= 8:
            return 'japan_summer'
        elif 9 <= date.month <= 11:
            return 'japan_fall'
        return 'japan_winter'
    
    # USA
    elif 'United States' in location or 'USA' in location:
        if 3 <= date.month <= 5:
            return 'usa_spring'
        elif 6 <= date.month <= 8:
            return 'usa_summer'
        elif 9 <= date.month <= 11:
            return 'usa_fall'
        return 'usa_winter'
    
    # Default case
    return 'indonesia_dry'  # default

def get_language(location):
    if 'Indonesia' in location:
        return 'id'
    elif 'Japan' in location or any('\u4e00' <= char <= '\u9fff' for char in location):
        # Check for Kanji characters in the location
        return 'ja'
    elif 'United States' in location or 'USA' in location:
        return 'en'
    return 'en'
