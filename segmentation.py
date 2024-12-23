customer_info = [
    'customer_id', 'customer_region', 
    'customer_age', 'is_repeat_customer'
]

temporal_data = [
    'first_order', 'last_order', 'days_between',
    *['HR_' + str(i) for i in range(24)], 
    *['DOW_' + str(i) for i in range(7)], 
    '0_7h', '8_14h', '15_19h', '20_23h'
]

product_vendor = [
    'vendor_count', 'product_count', 'is_chain'
]

spending_orders = [
    'total_orders', 'total_spend', 'avg_spend_prod',
    'promo_DELIVERY', 'promo_DISCOUNT', 'promo_FREEBIE', 'promo_NO DISCOUNT',
    'payment_method', 'pay_CARD', 'pay_CASH', 'pay_DIGI', 
    'payment_method_enc', 'last_promo_enc',
    *['HR_' + str(i) for i in range(24)], 
    *['DOW_' + str(i) for i in range(7)], 
]

cuisine_preferences = ['CUI_American',
                    'CUI_Asian',
                    'CUI_Beverages',
                    'CUI_Cafe',
                    'CUI_Chicken Dishes',
                    'CUI_Chinese',
                    'CUI_Desserts',
                    'CUI_Healthy',
                    'CUI_Indian',
                    'CUI_Italian',
                    'CUI_Japanese',
                    'CUI_Noodle Dishes',
                    'CUI_OTHER',
                    'CUI_Street Food / Snacks',
                    'CUI_Thai']