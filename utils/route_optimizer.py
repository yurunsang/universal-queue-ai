import pandas as pd

def optimize_route(predictions_df):
    """
    Simple heuristic: visit rides with shortest predicted waits first.
    Future enhancement: include walking distance (TSP).
    """
    if predictions_df.empty:
        return pd.DataFrame(columns=['order', 'ride', 'predicted_wait'])

    route = predictions_df.sort_values('predicted_wait').reset_index(drop=True)
    route['order'] = route.index + 1
    return route[['order', 'ride', 'predicted_wait']]