import { Check } from 'lucide-react'
import { Modal } from '@/components/ui/Modal'
import { Button } from '@/components/ui/Button'
import { UserForm } from './UserForm'
import type { CreateUserRequest } from '@/lib/types'

interface UserModalProps {
  open: boolean
  onClose: () => void
  form: CreateUserRequest
  errors: Partial<Record<keyof CreateUserRequest, string>>
  loading: boolean
  onFieldChange: (key: keyof CreateUserRequest, value: unknown) => void
  onSubmit: () => void
}

export const UserModal = ({
  open,
  onClose,
  form,
  errors,
  loading,
  onFieldChange,
  onSubmit
}: UserModalProps) => {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="✨ Add New User / إضافة مستخدم جديد"
      size="lg"
      footer={
        <>
          <Button variant="outline" onClick={onClose}>Cancel / إلغاء</Button>
          <Button variant="primary" onClick={onSubmit} loading={loading}>
            <Check className="w-4 h-4" /> Create Account / إنشاء الحساب
          </Button>
        </>
      }
    >
      <UserForm form={form} errors={errors} onChange={onFieldChange} />
    </Modal>
  )
}