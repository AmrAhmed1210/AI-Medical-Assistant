import { useEffect, useState } from 'react'
import { useDoctorStore } from '@/store/doctorStore'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { Star, MessageCircle } from 'lucide-react'
import toast from 'react-hot-toast'
import axiosInstance from '@/api/axiosInstance'
import type { ReviewDto } from '@/lib/types'

export default function DoctorReviews() {
  const { isLoadingProfile } = useDoctorStore()
  const [reviews, setReviews] = useState<ReviewDto[]>([])
  const [loading, setLoading] = useState(true)
  const [averageRating, setAverageRating] = useState(0)

  useEffect(() => {
    fetchReviews()
  }, [])

  const fetchReviews = async () => {
    try {
      setLoading(true)
      const response = await axiosInstance.get<ReviewDto[]>('/api/doctors/reviews')
      setReviews(response.data)
      
      if (response.data.length > 0) {
        const avg = response.data.reduce((sum: number, r: ReviewDto) => sum + r.rating, 0) / response.data.length
        setAverageRating(Math.round(avg * 10) / 10)
      }
    } catch (error) {
      console.error('Failed to fetch reviews:', error)
      toast.error('Failed to load reviews')
    } finally {
      setLoading(false)
    }
  }

  const renderStars = (rating: number) => (
    <div className="flex gap-1">
      {[...Array(5)].map((_, i) => (
        <Star
          key={i}
          size={16}
          className={i < rating ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'}
        />
      ))}
    </div>
  )

  const formatDate = (date: string | Date) => {
    return new Date(date).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-yellow-50 rounded-xl">
          <Star size={20} className="text-yellow-500" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-gray-800">Reviews & Ratings</h1>
          <p className="text-sm text-gray-500">Patient feedback about your practice</p>
        </div>
      </div>

      {isLoadingProfile || loading ? (
        <PageLoader />
      ) : (
        <>
          {reviews.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Overall Rating</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-4">
                  <div>
                    <div className="text-4xl font-bold text-yellow-500">{averageRating}</div>
                    <div className="text-sm text-gray-600">out of 5</div>
                  </div>
                  <div>
                    <div className="flex gap-1 mb-2">{renderStars(Math.round(averageRating))}</div>
                    <div className="text-sm text-gray-600">{reviews.length} review{reviews.length !== 1 ? 's' : ''}</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Patient Reviews</CardTitle>
            </CardHeader>
            <CardContent>
              {reviews.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <MessageCircle size={32} className="mx-auto mb-2 opacity-50" />
                  <p>No reviews yet. Great care leads to great reviews!</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {reviews.map((review) => (
                    <div key={review.id} className="pb-4 border-b border-gray-200 last:border-b-0">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <h3 className="font-semibold text-gray-800">
                            {review.patientName || 'Anonymous'}
                          </h3>
                          <p className="text-xs text-gray-500">
                            {formatDate(review.createdAt)}
                          </p>
                        </div>
                        <div className="flex gap-1">{renderStars(review.rating)}</div>
                      </div>
                      <p className="text-sm text-gray-700">{review.comment}</p>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}
