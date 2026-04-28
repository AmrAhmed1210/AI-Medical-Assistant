using MedicalAssistant.Domain.Entities;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IUnitOfWork : IAsyncDisposable
    {
        IPatientRepository Patients { get; }

        IAppointmentRepository Appointments { get; }

        IDoctorRepository Doctors { get; }

        IReviewRepository Reviews { get; }

        ISessionRepository Sessions { get; }

        IMessageRepository Messages { get; }

        IConsultationRepository Consultations { get; }

        IGenericRepository<TEntity> Repository<TEntity>() where TEntity : BaseEntity;

        /// <summary>
        /// Saves all changes to the database.
        /// </summary>
        Task<int> SaveChangesAsync();

        Task BeginTransactionAsync();
        Task CommitTransactionAsync();
        Task RollbackTransactionAsync();
    }
}