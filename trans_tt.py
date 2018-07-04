# coding=utf8
"""
使用方法：
1. python transform.py input_file_name output_file_name
"""
l_nm_list = ["is_audit_blocked","is_from_ningde","is_cluster","is_over_max_devices"
    ,"is_uid_blocked","is_bangsheng_blocked","is_agent","is_balance_lacking",
             "is_high_cheat","is_bill_not_first_hands","is_fake_bill_by_report",
             "is_tongdun_blocked","is_device_special","is_high_credit_level",
             "is_black_agent","is_black_list","is_operator_anonymous",
             "is_credit_city_changed","is_tongdun_multi_idcard_h3m",
             "is_zhima_v3_score","is_zhima_v3_multi_app_cur_day",
             "is_zhima_v3_multi_app_pre_week","is_zhima_v3_multi_app_pre_month",
             "is_zm_v3_fraud_score","is_zm_v3_basic_info_mismatch",
             "is_zm_v3_hit_fraud_watchlist","is_tencent_fraud_score",
             "is_tencent_credit_agent","is_tencent_fraud_risk_code",
             "is_baiqishi_blacklist","is_zhima_v3_industry_credit_overdue",
             "is_zhima_v3_industry_court_exec","is_zhima_v3_industry_payment_fraud",
             "is_zhima_v3_industry_hotel_default","is_suanhua_fraud_score",
             "is_tencent_fraud_suspect","is_coll_phone_decline",
             "is_tongdun_severe_multi_loan","is_credit_report_bad_status",
             "is_bill_inactive_pct_h6m","is_ind_high_limit_overdue",
             "is_smg3_high_risk_lbs","is_ind_user_info_invalid",
             "is_credit_bill_bad_status","is_student","f112","f007",
             "f114","f115","f116","f117","f118","f119","f303"]

def transform(input_file, output_file):
    records = read_records(input_file)
    with open(output_file, "w") as output_data:
        for record in records:
            processed_records = process_record(record)
            if processed_records is not None:
                for processed_record in processed_records:
                    output_line = processed_record
                    output_data.write(output_line + "\n")


def read_records(input_file):
    is_header = True
    with open(input_file) as input_data:
        for line in input_data:
            if is_header:
                is_header = False
            else:
                yield line.strip("\n")


def process_record(record):
    fileds = record.split(",")
    if len(fileds) < 59:    # 保持字段个数不会缺失
        return None
    forepart = ",".join(fileds[:5]) # 将不需要转化的前10列字段保留
    label_list = fileds[5:]
    encoded_num_arr = encode_list(label_list)
    result = []
    for elem in encoded_num_arr:
        result.append(forepart + "," + elem)
    return result


def encode_list(list_label):
    result = []
    for i in range(len(list_label)):
        if list_label[i] == 'True':
           result.append(l_nm_list[i])
    return result


if __name__ == '__main__':
    import sys
    input_data_file = sys.argv[1]
    output_data_file = sys.argv[2]
    transform(input_data_file, output_data_file)
