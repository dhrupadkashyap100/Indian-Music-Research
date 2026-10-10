
def adjust_tonic_for_augmentation(original_tonic, filename):
    """
    Adjusts the tonic frequency if the audio file has been pitch-shifted.
    
    Args:
        original_tonic (float): The ground truth tonic frequency.
        filename (str): The name of the audio file (e.g., 'song_pitchup.wav').
        
    Returns:
        float: The adjusted tonic frequency.
    """
    original_tonic = float(original_tonic)
    
    if "pitchup" in filename.lower():
        # Shifted up by 1 semitone
        return original_tonic * (2 ** (1 / 12.0))
    elif "pitchdown" in filename.lower():
        # Shifted down by 1 semitone
        return original_tonic * (2 ** (-1 / 12.0))
    
    # Speed changes or original don't affect pitch
    return original_tonic
