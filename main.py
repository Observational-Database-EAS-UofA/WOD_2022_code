from functions.read import read_WOD


def get_data():
    data_path = '/home/novaisc/workspace/obs_database/AW_CAA/CTD_DATA/WOD_2022/original_data/netcdf/ocldb1663004073.18805.CTD'
    save_path = '/home/novaisc/workspace/obs_database/AW_CAA/CTD_DATA/WOD_2022/ncfiles_raw'
    read_WOD(data_path, save_path)


if __name__ == "__main__":
    get_data()
