using MedicalAssistant.Domain.Entities;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IUnitOfWork : IAsyncDisposable
    {
        // --- مديول المرضى ---
        IPatientRepository Patients { get; }

        // --- مديول الحجوزات ---
        IAppointmentRepository Appointments { get; }

        // --- مديول الأطباء ---
        IDoctorRepository Doctors { get; }

        // --- مديول المراجعات ---
        IReviewRepository Reviews { get; }

        // --- مديول الجلسات ---
        ISessionRepository Sessions { get; }

        // --- مديول الرسائل ---
        IMessageRepository Messages { get; }

        // الوصول العام لأي مستودع
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
}