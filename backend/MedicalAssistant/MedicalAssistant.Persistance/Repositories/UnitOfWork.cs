using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities;
using MedicalAssistant.Domain.Entities.DoctorsModule;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore.Storage;
using System.Collections;

namespace MedicalAssistant.Persistance.Repositories;

public class UnitOfWork : IUnitOfWork
{
    private readonly MedicalAssistantDbContext _context;
    private IDbContextTransaction? _transaction;
    private Hashtable? _repositories;
    private bool _disposed;

    private IPatientRepository? _patients;
    private IAppointmentRepository? _appointments;
    private IDoctorRepository? _doctors;
    private IReviewRepository? _reviews;
    private IAdminRepository? _admins;

    public UnitOfWork(MedicalAssistantDbContext context)
    {
        _context = context;
    }

    public IPatientRepository Patients => _patients ??= new PatientRepository(_context);
    public IAppointmentRepository Appointments => _appointments ??= new AppointmentRepository(_context);
    public IDoctorRepository Doctors => _doctors ??= new DoctorRepository(_context);
    public IReviewRepository Reviews => _reviews ??= new ReviewRepository(_context);
    public IAdminRepository Admins => _admins ??= new AdminRepository(_context);

    public IGenericRepository<TEntity> Repository<TEntity>() where TEntity : BaseEntity
    {
        _repositories ??= new Hashtable();
        var typeName = typeof(TEntity).Name;

        if (!_repositories.ContainsKey(typeName))
        {
            var repositoryType = typeof(GenericRepository<>).MakeGenericType(typeof(TEntity));
            var repositoryInstance = Activator.CreateInstance(repositoryType, _context);
            _repositories.Add(typeName, repositoryInstance!);
        }
        return (IGenericRepository<TEntity>)_repositories[typeName]!;
    }

    public async Task<int> SaveChangesAsync() => await _context.SaveChangesAsync();

    public async Task BeginTransactionAsync() => _transaction = await _context.Database.BeginTransactionAsync();

    public async Task CommitTransactionAsync()
    {
        try
        {
            await _context.SaveChangesAsync();
            if (_transaction != null) await _transaction.CommitAsync();
        }
        catch { await RollbackTransactionAsync(); throw; }
        finally { await DisposeTransactionAsync(); }
    }

    public async Task RollbackTransactionAsync()
    {
        if (_transaction != null)
        {
            await _transaction.RollbackAsync();
            await DisposeTransactionAsync();
        }
    }

    private async Task DisposeTransactionAsync()
    {
        if (_transaction != null) { await _transaction.DisposeAsync(); _transaction = null; }
    }

    public async ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            if (_transaction != null) await DisposeTransactionAsync();
            await _context.DisposeAsync();
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    public void Dispose()
    {
        if (!_disposed) { _transaction?.Dispose(); _context.Dispose(); _disposed = true; }
        GC.SuppressFinalize(this);
    }
}