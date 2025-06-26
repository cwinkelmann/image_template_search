from pathlib import Path

p_source = Path("/Volumes/IFA_ANDREA/Raw photos all years")
destination = Path("/Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y")


for p in p_source.glob("*"):
    if not p.is_dir():
        continue
    if p.name.startswith("."):
        continue
    rsync_command = f"rsync -avz '{p}' {destination}/ "
    print(rsync_command)


"""


rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Fernandina' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Floreana' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Genovesa' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Isabela' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Marchena' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Pinta' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/San Cristobal' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Santa Fe' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Wolf' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Española' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Bainbridge Rocks' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Bartolome' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Beagles' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Pinzón' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Plazas' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Daphnes' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Santiago' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Santa Cruz' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Rabida' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/Seymour Norte' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 


"""


"""
rsync -avz '/Volumes/IFA_ANDREA/Raw photos all years/El Nino project' /Volumes/G-DRIVE/Iguanas_From_Above/raw_photos_all_y/ 
"""