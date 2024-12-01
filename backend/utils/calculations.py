def calculate_report(data):
    width_fakel = 0.1
    layers = 2
    paint_consumption_per_layer = 0.3
    labor_time_per_layer = 0.1

    total_area = sum([el['area'] for el in data['elements']])
    total_paint = total_area * layers * paint_consumption_per_layer
    total_time = total_area * layers * labor_time_per_layer

    report = {
        'total_area': total_area,
        'total_paint': total_paint,
        'total_time': total_time
    }
    return report
