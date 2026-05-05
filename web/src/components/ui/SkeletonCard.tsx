import { Card } from './Card'
import { Skeleton } from './Skeleton'

interface SkeletonCardProps {
  count?: number
  className?: string
}

export function SkeletonCard({ count = 1, className = '' }: SkeletonCardProps) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i} className={`p-4 ${className}`}>
          <div className="space-y-3">
            <Skeleton className="h-4 w-1/3" />
            <Skeleton className="h-6 w-2/3" />
            <Skeleton className="h-4 w-1/2" />
          </div>
        </Card>
      ))}
    </>
  )
}
