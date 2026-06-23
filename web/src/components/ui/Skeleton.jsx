import { cx } from './utils'

export default function Skeleton({ className }) {
  return <div className={cx('animate-pulse rounded-lg bg-white/10', className)} />
}
