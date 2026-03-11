using MedicalAssistant.Domain.Entities;
using MedicalAssistant.Domain.Entities.DoctorsModule;

namespace MedicalAssistant.Domain.Contracts;

/// <summary>
/// Unit of Work interface.
/// Used for managing transactions and ensuring all operations execute as a single unit.
/// </summary>
public interface IUnitOfWork : IAsyncDisposable // تم التحديث لـ IAsyncDisposable لعمليات الـ Async الحديثة
{
    // --- مديول المرضى ---
    IPatientRepository Patients { get; }

    // --- مديول الحجوزات ---
    IAppointmentRepository Appointments { get; }

    // --- مديول الأطباء (تمت الإضافة لخدمة الـ DoctorService) ---
    IDoctorRepository Doctors { get; }

    // --- مديول المراجعات (مطلوب في مستند المتطلبات) ---
    // IReviewRepository Reviews { get; } // يمكنك فك التعليق بعد إنشاء الـ Repository الخاص به

    // الوصول العام لأي مستودع (Generic Repository Access)
    IGenericRepository<TEntity> Repository<TEntity>() where TEntity : BaseEntity;

    /// <summary>
    /// Saves all changes to the database.
    /// </summary>
    Task<int> SaveChangesAsync();

    // --- إدارة العمليات (Transaction Management) ---
    Task BeginTransactionAsync();
    Task CommitTransactionAsync();
    Task RollbackTransactionAsync();
}