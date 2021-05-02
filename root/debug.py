import pandas as pd
if __name__ == "__main__":
    raw_data = pd.read_json('../anilist.json')
    raw_data = raw_data.T

    animes_with_AnimationStudio = 0

    for anime_id in raw_data.index:

        has_AnimationStudio = False
        for studio in raw_data.loc[anime_id].studios['nodes']:
            if studio['isAnimationStudio']:
                has_AnimationStudio = True
                break
        if not has_AnimationStudio:
            print(anime_id)
        animes_with_AnimationStudio += has_AnimationStudio

    print(f'Animes Total: {len(raw_data.index)}')
    print(f'Animes with AnimationStudio: {animes_with_AnimationStudio}')