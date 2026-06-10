#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_add_mul_insecure() {
        let a: Fr = 42u128;
        let b: Fr = 7u128;
        let sa = share_secret(a);
        let sb = share_secret(b);
        let sum = add_shares(&sa, &sb);
        assert_eq!(reconstruct(&sum), a.wrapping_add(b));
        let prod = mul_insecure(&sa, &sb);
        assert_eq!(reconstruct(&prod), a.wrapping_mul(b));
    }

    #[test]
    fn test_mul_ab3_simulate() {
        let a: Fr = 5u128;
        let b: Fr = 11u128;
        let sa = share_secret(a);
        let sb = share_secret(b);
        let prod = mul_ab3(&sa, &sb);
        assert_eq!(reconstruct(&prod), a.wrapping_mul(b));
        let prod2 = mul_ab3_simulate(&sa, &sb);
        assert_eq!(reconstruct(&prod2), a.wrapping_mul(b));
    }

    #[test]
    fn test_bits_ops() {
        let t = share_bool(true);
        let f = share_bool(false);
        let andv = and(&t, &f);
        assert_eq!(reconstruct_bit(&andv), true & false);
        let orv = or(&t, &f);
        assert_eq!(reconstruct_bit(&orv), true | false);
        let notv = not(&t);
        assert_eq!(reconstruct_bit(&notv), !true);
    }
}
