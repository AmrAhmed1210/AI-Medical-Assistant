using System.Linq.Expressions;
using MedicalAssistant.Domain.Entities;

namespace MedicalAssistant.Domain.Contracts
{
    /// <summary>
    /// Generic repository interface.
    /// Contains basic CRUD operations for database entities.
    /// </summary>
    /// <typeparam name="T">Entity type to work with</typeparam>
    public interface IGenericRepository<T> where T : BaseEntity
    {
        /// <summary>
        /// Returns all entities.
        /// </summary>
        Task<IEnumerable<T>> GetAllAsync();

        /// <summary>
        /// Returns all entities with included navigation properties.
        /// </summary>
        Task<IEnumerable<T>> GetAllAsync(params Expression<Func<T, object>>[] includes);

        /// <summary>
        /// Returns an entity by id.
        /// </summary>
        Task<T?> GetByIdAsync(int id);

        /// <summary>
        /// Returns an entity by id with included navigation properties.
        /// </summary>
        Task<T?> GetByIdAsync(int id, params Expression<Func<T, object>>[] includes);

        /// <summary>
        /// Finds entities matching a predicate.
        /// </summary>
        Task<IEnumerable<T>> FindAsync(Expression<Func<T, bool>> predicate);

        /// <summary>
        /// Finds entities matching a predicate with included navigation properties.
        /// </summary>
        Task<IEnumerable<T>> FindAsync(Expression<Func<T, bool>> predicate, params Expression<Func<T, object>>[] includes);

        /// <summary>
        /// Adds a new entity.
        /// </summary>
        Task AddAsync(T entity);

        /// <summary>
        /// Adds a range of entities.
        /// </summary>
        Task AddRangeAsync(IEnumerable<T> entities);

        /// <summary>
        /// Updates an existing entity.
        /// </summary>
        void Update(T entity);

        /// <summary>
        /// Deletes an entity.
        /// </summary>
        void Delete(T entity);

        /// <summary>
        /// Deletes a range of entities.
        /// </summary>
        void DeleteRange(IEnumerable<T> entities);

        /// <summary>
        /// Checks if any entity matches a predicate.
        /// </summary>
        Task<bool> ExistsAsync(Expression<Func<T, bool>> predicate);

        /// <summary>
        /// Returns total count of entities.
        /// </summary>
        Task<int> CountAsync();

        /// <summary>
        /// Returns count of entities matching a predicate.
        /// </summary>
        Task<int> CountAsync(Expression<Func<T, bool>> predicate);
    }
}
